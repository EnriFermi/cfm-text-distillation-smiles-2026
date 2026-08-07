"""Print what each of the three TinyStories checkpoints actually is."""

import sys
from pathlib import Path

import torch

CFM = Path("/home/andrey/Documents/projects/SMILES 2026/2 models/cfm_mod/last.ckpt")
BD3 = Path("/home/andrey/Documents/projects/SMILES 2026/2 models/bd3lm/last.pt")
MDLM = Path(
    "/home/andrey/Documents/projects/SMILES 2026/2 models/"
    "mdlm_tinystories_gpt2_h384_l6_seed12345/checkpoints/last.pt"
)


def n_params(state):
    return sum(v.numel() for v in state.values() if torch.is_tensor(v))


print("=" * 70, "\nCFM:", CFM)
ck = torch.load(CFM, map_location="cpu", weights_only=False)
print(" top-level keys:", sorted(ck.keys()))
print(" global_step:", ck.get("global_step"), " epoch:", ck.get("epoch"))
hp = ck.get("hyper_parameters") or {}
print(" hyper_parameters keys:", sorted(hp.keys()))
for k, v in hp.items():
    if not isinstance(v, (dict, list)):
        print(f"   {k} = {v!r}"[:200])
sd = ck["state_dict"]
net = {k[len("net."):].replace("_orig_mod.", ""): v for k, v in sd.items() if k.startswith("net.")}
print(" n params (state_dict):", f"{n_params(sd):,}", " net-only:", f"{n_params(net):,}")
for k in list(net)[:6]:
    print("   ", k, tuple(net[k].shape))
print(" output_layer:", tuple(net["output_layer.linear.weight"].shape))
print(" rotary cos_cached:", tuple(net["rotary_emb.cos_cached"].shape))
nb = 1 + max(int(k.split(".")[1]) for k in net if k.startswith("blocks."))
print(" n_blocks:", nb, " s_map:", tuple(net["s_map.mlp.0.weight"].shape))
print(" has final_norm (rms embed):", any("final_norm" in k for k in net))
del ck, sd, net

print("=" * 70, "\nBD3LM:", BD3)
pl = torch.load(BD3, map_location="cpu", weights_only=False)
print(" top-level keys:", sorted(pl.keys()))
for k in ("step", "global_step", "epoch"):
    if k in pl:
        print(f" {k}:", pl[k])
cfg = pl["config"]
import json
print(" config:", json.dumps(cfg if isinstance(cfg, dict) else dict(cfg), indent=1, default=str)[:4000])
m = pl.get("ema_model") or pl["model"]
print(" weights available: model=%s ema_model=%s" % ("model" in pl, "ema_model" in pl))
print(" n params:", f"{n_params(m):,}")
del pl, m

print("=" * 70, "\nMDLM:", MDLM)
pl = torch.load(MDLM, map_location="cpu", weights_only=False)
print(" top-level keys:", sorted(pl.keys()))
for k in ("step", "global_step", "epoch"):
    if k in pl:
        print(f" {k}:", pl[k])
cfg = pl["config"]
print(" config:", json.dumps(cfg if isinstance(cfg, dict) else dict(cfg), indent=1, default=str)[:4000])
m = pl.get("ema_model") or pl["model"]
print(" weights available: model=%s ema_model=%s" % ("model" in pl, "ema_model" in pl))
print(" n params:", f"{n_params(m):,}")
