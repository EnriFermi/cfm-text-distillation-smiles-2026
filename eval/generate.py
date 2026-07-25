"""Model loading + sample generation, dispatched by sampler config.

Reused by ``eval/run_eval.py`` and available for notebooks/ad-hoc scripts. Kept free
of Hydra ``@main`` so it composes anywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor

from block.sampling import block_causal_sample, blockwise_sample
from semicat.models.textsemicat import TextSemicatModule


def load_module(cfg: DictConfig, device: str) -> TextSemicatModule:
    """Instantiate the model from config and load weights if ``cfg.ckpt_path`` is set.

    Instantiating the *whole* module (net included) from ``cfg.model`` — rather than
    ``load_from_checkpoint`` — keeps this class-agnostic (works for M1/M2's
    ``TextSemicatModule`` and M3's ``BlockSemicatModule`` alike). Loading is
    Loading is strict except for deterministic RoPE buffers
    (``net.rotary_emb.{cos,sin}_cached``), which newer checkpoints may omit.
    Any other missing/unexpected keys still raise — mismatched ``model.net.*``
    would otherwise leave random weights and score garbage.

    With ``ckpt_path=null`` you get a fresh (untrained) model — enough to smoke-test the
    sampling/eval plumbing without a checkpoint.
    """
    module = hydra.utils.instantiate(cfg.model)
    if cfg.get("ckpt_path"):
        ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # undo the torch.compile prefix (net._orig_mod.) if present
        state = {k.replace("net._orig_mod.", "net."): v for k, v in state.items()}
        # RoPE cos/sin tables are deterministic from (length, dim) and may be omitted
        # from newer checkpoints (non-persistent buffers / recompute-on-init).
        _ok_missing = {
            "net.rotary_emb.cos_cached",
            "net.rotary_emb.sin_cached",
        }
        incompatible = module.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [k for k in incompatible.missing_keys if k not in _ok_missing]
        if unexpected or missing:
            raise RuntimeError(
                "Error(s) in loading state_dict: "
                f"Missing key(s): {missing}; Unexpected key(s): {unexpected}"
            )
    return module.to(device).eval()


def make_datamodule(cfg: DictConfig, device: str):
    """Set up the datamodule for detok / gold sequences without a Lightning Trainer.

    ``Text8DataModule.setup`` reads ``self.trainer`` only for the device and
    world_size, so we hand it a tiny shim rather than spin up a Trainer.
    """
    dm = hydra.utils.instantiate(cfg.data)
    dm.trainer = SimpleNamespace(
        strategy=SimpleNamespace(root_device=torch.device(device)), world_size=1
    )
    dm.setup()
    return dm


def gold_sequences(dm, n: int, device: str) -> Tensor:
    """``n`` clean test sequences spread evenly over the test set.

    ``Text8Dataset`` windows have stride 1, so consecutive indices are near-duplicates;
    even spacing decorrelates the sample.
    """
    ds = dm.data_test
    idx = torch.linspace(0, len(ds) - 1, n).round().long().tolist()
    return torch.stack([ds[i] for i in idx]).to(device)


@torch.inference_mode()
def sample_tokens(
    module: TextSemicatModule,
    sampler: DictConfig,
    *,
    n_samples: int,
    batch_size: int,
    length: int,
    gold_prefix: Tensor | None = None,
) -> Tensor:
    """Generate ``n_samples`` sequences under the given sampler config.

    Dispatches on ``sampler.name``: ``full_cfm`` (M1, full-sequence),
    ``bcfm_infer`` (M2, blockwise + clean-prefix KV cache), or ``bcfm_train`` (M3).
    Returns ``(n_samples, length)`` on CPU.
    """
    out, done = [], 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        if sampler.name == "full_cfm":
            z = module.sample_flow_map_batch(batch_size=bs, sampling_steps=sampler.steps)
            toks = z.argmax(dim=-1)
        elif sampler.name == "bcfm_infer":
            gp = gold_prefix[done:done + bs] if gold_prefix is not None else None
            sched = [tuple(st) for st in sampler.schedule] if sampler.get("schedule") else None
            num_blocks = sampler.get("num_blocks")
            toks = blockwise_sample(
                module,
                sampler.block_size,
                sampler.steps_per_block,
                batch_size=bs,
                length=length if num_blocks is None else None,
                num_blocks=int(num_blocks) if num_blocks is not None else None,
                discretize=sampler.discretize,
                schedule=sched,
                gold_prefix=gp,
            )
        elif sampler.name == "bcfm_train":
            sched = [tuple(st) for st in sampler.schedule] if sampler.get("schedule") else None
            toks = block_causal_sample(
                module,
                sampler.block_size,
                sampler.steps_per_block,
                batch_size=bs,
                length=length,
                discretize=sampler.discretize,
                schedule=sched,
                use_kv_cache=bool(sampler.get("use_kv_cache", True)),
            )
        else:
            raise ValueError(f"unknown sampler '{sampler.name}'")
        out.append(toks.cpu())
        done += bs
    return torch.cat(out)[:n_samples]
