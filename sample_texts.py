"""Sanity check: print a couple of real samples from each checkpoint.

Latency is only meaningful if the thing being timed actually generates text, so
this decodes a few sequences from both models with the same GPT-2 tokenizer both
were trained on.

    python "eval test/sample_texts.py" --n 2 --cfm-nfe 4 --bd3-steps 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from common import BD3_CKPT, CFM_CKPT, HERE

CFM_SNIPPET = '''
import torch, sys
sys.path.insert(0, {shim!r}); sys.path.insert(0, {repo!r})
from eval.dump_samples import build_module, detect_arch
from block.sampling import block_causal_sample
import transformers

ck = torch.load({ckpt!r}, map_location="cpu", weights_only=False)
arch = detect_arch(ck); arch["_path"] = {ckpt!r}; del ck
module = build_module(arch, "cuda")
torch.manual_seed(0)
ids = block_causal_sample(module, block_size=arch["block_size"],
                          steps_per_block={nfe}, batch_size={n},
                          discretize="argmax")
tok = transformers.AutoTokenizer.from_pretrained("gpt2")
for i, text in enumerate(tok.batch_decode(ids, skip_special_tokens=True)):
    print(f"--- CFM sample {{i}} (nfe={nfe}) ---")
    print(text[:600])
'''

BD3_SNIPPET = '''
import torch, sys
sys.path.insert(0, {repo!r})
from bcfm_baselines.checkpoint import load_checkpoint_model
from bcfm_baselines.models.generative_text import GenerationConfig
import transformers

model, payload = load_checkpoint_model({ckpt!r}, "cuda")
sampling = dict(payload["config"]["sampling"])
sampling.update(batch_size={n}, steps_per_block={steps}, first_hitting=False)
result = model.generate(GenerationConfig(**sampling))
tok = transformers.AutoTokenizer.from_pretrained("gpt2")
for i, text in enumerate(tok.batch_decode(result.tokens, skip_special_tokens=True)):
    print(f"--- BD3-LM sample {{i}} (steps/block={steps}) ---")
    print(text[:600])
'''


def run(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], cwd=HERE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--cfm-nfe", type=int, default=4)
    parser.add_argument("--bd3-steps", type=int, default=4)
    args = parser.parse_args()

    from common import BD3_REPO, CFM_REPO

    run(CFM_SNIPPET.format(shim=str(HERE / "shim"), repo=str(CFM_REPO),
                           ckpt=str(CFM_CKPT), nfe=args.cfm_nfe, n=args.n))
    run(BD3_SNIPPET.format(repo=str(BD3_REPO), ckpt=str(BD3_CKPT),
                           steps=args.bd3_steps, n=args.n))


if __name__ == "__main__":
    main()
