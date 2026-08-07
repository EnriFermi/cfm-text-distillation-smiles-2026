"""Standalone exact-semantics incremental M3 sampler prototype.

No project source is modified. The prototype computes only the current noisy block,
and incrementally caches clean-stream K/V for finalized blocks.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from block.sampling import block_causal_sample, uniform_schedule
from eval.dump_samples import build_module, detect_arch
from semicat.net.duo import modulate_fused, split_and_apply_rotary_pos_emb

OUT = Path(__file__).resolve().parent
CKPT = ROOT / "logs/train/2026-07-28_12-43-24_12345_local/checkpoints/step_0100000.ckpt"


def apply_rotary_block(qkv, rotary_cos_sin, lo, hi):
    cos, sin = rotary_cos_sin
    sliced = (cos[:, lo:hi].to(qkv.dtype), sin[:, lo:hi].to(qkv.dtype))
    return split_and_apply_rotary_pos_emb(qkv, sliced)


def qkv_block(net, layer, h, rotary_cos_sin, lo, hi):
    qkv = layer.attn_qkv(h).reshape(
        h.shape[0], h.shape[1], 3, net.n_heads, net.hidden_size // net.n_heads
    )
    return apply_rotary_block(qkv, rotary_cos_sin, lo, hi)


def attend(q, k, v):
    # K contains exactly the keys allowed by the block-causal mask, so no mask is needed.
    y = F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
        attn_mask=None, dropout_p=0.0, is_causal=False,
    )
    return y.transpose(1,2).reshape(q.shape[0], q.shape[1], -1)


def block_forward(net, x_block, s, t, lo, clean_k, clean_v, *, return_kv=False):
    """Block hidden/logits; clean_k/v are the strictly prior finalized prefix."""
    cond = net._condition(s, t)
    h = net._embed(x_block, s, cond)
    rotary = net.rotary_emb()
    new_ks=[]; new_vs=[]
    for idx, layer in enumerate(net.blocks):
        modulation = net._layer_modulation(layer, cond)
        shift_msa, scale_msa, *_ = modulation
        residual = h
        h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
        q,k,v = qkv_block(net, layer, h_norm, rotary, lo, lo+h.shape[1])
        full_k = torch.cat((clean_k[idx], k), dim=1) if clean_k[idx] is not None else k
        full_v = torch.cat((clean_v[idx], v), dim=1) if clean_v[idx] is not None else v
        h = net._finish_layer(layer, residual, attend(q,full_k,full_v), modulation)
        if return_kv:
            new_ks.append(k); new_vs.append(v)
    if return_kv:
        return new_ks,new_vs
    return net._final_layer(h,cond)


COMPILED_NOISY = None
COMPILED_CLEAN = None


def compiled_noisy_forward(net, x_block, s, t, lo, clean_k, clean_v):
    global COMPILED_NOISY
    if COMPILED_NOISY is None:
        COMPILED_NOISY = torch.compile(
            block_forward, fullgraph=True, mode="reduce-overhead"
        )
    # reduce-overhead uses static CUDA-graph output buffers.  Own the logits before
    # the next replay so a later jump cannot overwrite a still-live endpoint.
    return COMPILED_NOISY(net,x_block,s,t,lo,clean_k,clean_v).clone()


def clean_forward_only(net, x_block, s, t, lo, clean_k, clean_v):
    return block_forward(net,x_block,s,t,lo,clean_k,clean_v,return_kv=True)


def compiled_clean_forward(net, x_block, s, t, lo, clean_k, clean_v):
    global COMPILED_CLEAN
    if COMPILED_CLEAN is None:
        COMPILED_CLEAN = torch.compile(
            clean_forward_only, fullgraph=True, mode="reduce-overhead"
        )
    return COMPILED_CLEAN(net,x_block,s,t,lo,clean_k,clean_v)


@torch.inference_mode()
def incremental_sample(module, nfe, *, discretize="argmax", batch_size=1,
                       initial_noise=None, return_endpoints=False, compiled=False):
    net=module.net; L=net.length; B=net.block_size; K=net.vocab_size
    sched=uniform_schedule(nfe)
    tokens=torch.zeros(batch_size,L,dtype=torch.long,device=module.device)
    clean_k=[None]*len(net.blocks); clean_v=[None]*len(net.blocks)
    endpoints=[]
    ones=torch.ones(batch_size,B,device=module.device)
    for b in range(L//B):
        lo=b*B
        z = (initial_noise[b].clone() if initial_noise is not None
             else module.prior((batch_size,B,K),device=module.device))
        for s_value,t_value in sched:
            s=torch.full((batch_size,B),s_value,device=module.device)
            t=torch.full((batch_size,B),t_value,device=module.device)
            logits=(compiled_noisy_forward(net,z,s,t,lo,clean_k,clean_v) if compiled
                    else block_forward(net,z,s,t,lo,clean_k,clean_v))
            q=logits.softmax(-1)
            z=z+((t_value-s_value)/(1.0-s_value+1e-8))*(q-z)
        endpoints.append(z)
        if discretize=="argmax": block=z.argmax(-1)
        else:
            block=torch.multinomial(z.clamp_min(0).flatten(0,1),1).squeeze(-1).view(batch_size,B)
        tokens[:,lo:lo+B]=block
        if b+1 < L//B:
            clean=F.one_hot(block,K).to(z.dtype)
            new_k,new_v=(compiled_clean_forward(net,clean,ones,ones,lo,clean_k,clean_v)
                         if compiled else block_forward(net,clean,ones,ones,lo,clean_k,clean_v,return_kv=True))
            if compiled:
                # Compiled reduce-overhead mode returns static CUDAGraph buffers.
                new_k=[value.clone() for value in new_k]
                new_v=[value.clone() for value in new_v]
            for i in range(len(clean_k)):
                clean_k[i]=new_k[i] if clean_k[i] is None else torch.cat((clean_k[i],new_k[i]),1)
                clean_v[i]=new_v[i] if clean_v[i] is None else torch.cat((clean_v[i],new_v[i]),1)
    return (tokens,endpoints) if return_endpoints else tokens


def med_ms(fn,warmup=3,repeats=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); times=[]
    for _ in range(repeats):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); times.append(a.elapsed_time(b))
    return statistics.median(times),times


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--nfe",nargs="+",type=int,default=[1,2,4,8,16,32])
    parser.add_argument("--repeats",type=int,default=10)
    parser.add_argument("--compiled", action="store_true"); args=parser.parse_args()
    raw=torch.load(CKPT,map_location="cpu",weights_only=False); arch=detect_arch(raw); arch["_path"]=CKPT
    print(f"[config] checkpoint={CKPT} step={raw.get('global_step')} device=cuda dtype=fp32 seed=0 arch={arch} out={OUT}",flush=True)
    module=build_module(arch,"cuda")
    rows=[]
    for nfe in args.nfe:
        print(f"[stage=benchmark] nfe={nfe} current",flush=True)
        torch.manual_seed(123); torch.cuda.manual_seed_all(123)
        current,cur_times=med_ms(lambda:block_causal_sample(module,arch['block_size'],nfe,batch_size=1,discretize='argmax'),repeats=args.repeats)
        print(f"[stage=benchmark] nfe={nfe} incremental",flush=True)
        torch.manual_seed(123); torch.cuda.manual_seed_all(123)
        incr,inc_times=med_ms(lambda:incremental_sample(module,nfe,compiled=args.compiled),repeats=args.repeats)
        row={"nfe":nfe,"current_median_ms":current,"incremental_median_ms":incr,
             "speedup":current/incr,"compiled":args.compiled,
             "current_times_ms":cur_times,"incremental_times_ms":inc_times}
        rows.append(row); print("[metric] "+json.dumps(row),flush=True)
        (OUT/"prototype_benchmark.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
    print(f"[done] artifact={OUT/'prototype_benchmark.json'}",flush=True)

if __name__=="__main__": main()
