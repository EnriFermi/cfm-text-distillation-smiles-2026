"""CPU discriminator: full doubled masked forward vs incremental exact-KV forward."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))

from block.block_dit import BlockDIT
from prototype_incremental import block_forward


def main():
    torch.manual_seed(7)
    net=BlockDIT(vocab_size=17,hidden_size=24,cond_dim=8,n_blocks=2,n_heads=3,
                 dropout=0.0,length=8,block_size=2,embed_type="rms",
                 attention_backend="sdpa").eval()
    batch,L,B,K=2,8,2,17
    tokens=torch.randint(0,K,(batch,L))
    clean=F.one_hot(tokens,K).float()
    clean_k=[None]*len(net.blocks); clean_v=[None]*len(net.blocks)
    ones_block=torch.ones(batch,B)
    rows=[]
    for b in range(L//B):
        lo=b*B; hi=lo+B
        z=torch.randn(batch,B,K)
        s_value,t_value=0.25,0.75
        noisy=torch.zeros(batch,L,K); noisy[:,lo:hi]=z
        s_tok=torch.zeros(batch,L); t_tok=torch.zeros(batch,L)
        s_tok[:,lo:hi]=s_value; t_tok[:,lo:hi]=t_value
        ones=torch.ones(batch,L)
        with torch.inference_mode():
            ref=net(torch.cat((clean,noisy),1),torch.cat((ones,s_tok),1),
                    torch.cat((ones,t_tok),1))[:,L+lo:L+hi]
            got=block_forward(net,z,torch.full((batch,B),s_value),
                              torch.full((batch,B),t_value),lo,clean_k,clean_v)
        diff=(ref-got).abs()
        rows.append({"block":b,"max_abs":diff.max().item(),
                     "mean_abs":diff.mean().item(),
                     "argmax_agreement":(ref.argmax(-1)==got.argmax(-1)).float().mean().item()})
        if b+1<L//B:
            with torch.inference_mode():
                nk,nv=block_forward(net,clean[:,lo:hi],ones_block,ones_block,lo,
                                    clean_k,clean_v,return_kv=True)
            for i in range(len(clean_k)):
                clean_k[i]=nk[i] if clean_k[i] is None else torch.cat((clean_k[i],nk[i]),1)
                clean_v[i]=nv[i] if clean_v[i] is None else torch.cat((clean_v[i],nv[i]),1)
    payload={"seed":7,"device":"cpu","dtype":"float32","rows":rows,
             "overall_max_abs":max(r["max_abs"] for r in rows),
             "all_argmax_equal":all(r["argmax_agreement"]==1.0 for r in rows)}
    out=Path(__file__).with_name("incremental_semantics_cpu.json")
    out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
