"""Profile the current M3 batch=1 inference path without modifying project code."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from eval.dump_samples import build_module, detect_arch
from block.sampling import block_causal_sample


OUT = Path(__file__).resolve().parent
CKPT = ROOT / "logs/train/2026-07-28_12-43-24_12345_local/checkpoints/step_0100000.ckpt"


def sync():
    torch.cuda.synchronize()


def median_ms(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        times.append(start.elapsed_time(end))
    return float(torch.tensor(times).median()), times


def main():
    print(f"[config] checkpoint={CKPT} device=cuda dtype=float32 seed=0 out={OUT}", flush=True)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    arch = detect_arch(raw); arch["_path"] = CKPT
    print(f"[stage=model_build] arch={arch}", flush=True)
    module = build_module(arch, "cuda")
    net = module.net
    L, K, B = arch["length"], arch["vocab_size"], arch["block_size"]
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    b = 8; lo=b*B; hi=lo+B
    tokens = torch.randint(0, K, (1,L), device="cuda")
    z_blk = module.prior((1,B,K), device=module.device)
    clean = torch.nn.functional.one_hot(tokens, K).to(z_blk.dtype)
    noisy = torch.zeros(1,L,K,device="cuda",dtype=z_blk.dtype); noisy[:,lo:hi]=z_blk
    s_tok=torch.zeros(1,L,device="cuda"); t_tok=torch.zeros_like(s_tok)
    t_tok[:,lo:hi]=1.0
    ones=torch.ones(1,L,device="cuda")
    x=torch.cat([clean,noisy],1); s_full=torch.cat([ones,s_tok],1); t_full=torch.cat([ones,t_tok],1)

    with torch.inference_mode():
        med_forward, raw_forward = median_ms(lambda: net(x,s_full,t_full), repeats=30)
        med_seq1, raw_seq1 = median_ms(lambda: block_causal_sample(module,B,1,batch_size=1,discretize="argmax"), warmup=3,repeats=10)

        print("[stage=profiler] one full doubled forward", flush=True)
        for _ in range(3): net(x,s_full,t_full)
        sync()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                     profile_memory=True, record_shapes=True, with_stack=False) as prof:
            with record_function("m3_current_full_forward"):
                net(x,s_full,t_full)
        sync()
        (OUT/"current_forward_table.txt").write_text(
            prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=80)+"\n")
        prof.export_chrome_trace(str(OUT/"current_forward_trace.json"))

        print("[stage=profiler] full NFE=1 sequence", flush=True)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                     profile_memory=True, record_shapes=True, with_stack=False) as seq_prof:
            with record_function("m3_current_sequence_nfe1"):
                block_causal_sample(module,B,1,batch_size=1,discretize="argmax")
        sync()
        (OUT/"current_sequence_nfe1_table.txt").write_text(
            seq_prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=100)+"\n")
        seq_prof.export_chrome_trace(str(OUT/"current_sequence_nfe1_trace.json"))

    summary = {
        "checkpoint": str(CKPT), "global_step": raw.get("global_step"),
        "gpu": torch.cuda.get_device_name(0), "dtype":"float32", "batch_size":1,
        "length":L,"block_size":B,"vocab_size":K,"hidden_size":arch["hidden_size"],
        "n_layers":arch["n_blocks"],"n_heads":arch["n_heads"],
        "one_forward_median_ms":med_forward,"one_forward_times_ms":raw_forward,
        "sequence_nfe1_median_ms":med_seq1,"sequence_nfe1_times_ms":raw_seq1,
        "forwards_per_sequence_nfe1":L//B,
        "dense_input_elements_per_forward":2*L*K,
        "dense_output_logits_elements_per_forward":2*L*K,
        "used_output_logits_elements_per_forward":B*K,
    }
    (OUT/"current_profile_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("[done] "+json.dumps(summary,indent=2),flush=True)

if __name__ == "__main__": main()
