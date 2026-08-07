"""Entropy, generative perplexity, and MAUVE for generated text.

``compute_mean_entropy`` and ``compute_mean_gen_ppl`` are deliberate line-for-line
ports of the CFM reference implementation (``semicat/metric/text_dist.py``, itself
derived from the Duo evaluation pipeline).  The two sides of the comparison have
to be scored by exactly the same judge under exactly the same masking rule, so
the tokenizer, the ``gpt2-large`` scorer, the padding/truncation behaviour, and
the "all content tokens plus the first EOS" token mask are reproduced as-is --
including the tail batch that the reference drops when the sample count is not a
multiple of ``batch_size``.  Keep sample counts divisible by the scoring batch
size and nothing is dropped.

MAUVE has no counterpart in the reference repository; it follows the canonical
``mauve-text`` recipe (``gpt2-large`` features, scaling factor 5) so the numbers
are comparable with the published literature.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_PPL_MODEL = "gpt2-large"
DEFAULT_MAUVE_MODEL = "gpt2-large"


class TextMetrics:
    """Metrics on a batch of generated strings (entropy, Gen-PPL, MAUVE)."""

    @classmethod
    def _load_tokenizer(cls, tokenizer_model: str = DEFAULT_PPL_MODEL):
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @classmethod
    def _retokenize(
        cls,
        tokenizer,
        max_length: int,
        text_samples: list[str],
        device: str,
    ) -> tuple[Tensor, Tensor]:
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "return_attention_mask": True,
            "truncation": True,
            "padding": True,
            "max_length": max_length,
        }
        samples = tokenizer(text_samples, **tokenizer_kwargs)
        return samples["input_ids"].to(device), samples["attention_mask"].to(device)

    @classmethod
    def compute_mean_entropy(cls, tokens: list[Tensor]) -> float:
        """Unigram entropy over token IDs, averaged across sample batches."""
        entropies = []
        for batch in tokens:
            _, counts = batch.unique(return_counts=True, sorted=False)
            entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
            entropies += [entropy]
        return float(np.mean(entropies))

    @classmethod
    def compute_mean_gen_ppl(
        cls,
        text_samples: list[str],
        batch_size: int,
        context_size: int = 1024,
        ppl_model: str = DEFAULT_PPL_MODEL,
        device: str = "cuda",
    ) -> float:
        """Generative perplexity of `text_samples` under a pre-trained `ppl_model`."""
        import transformers

        with torch.inference_mode():
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            ppls = []
            effective_size = 0
            tokenizer = cls._load_tokenizer()
            model = transformers.AutoModelForCausalLM.from_pretrained(ppl_model).eval()
            model = model.to(device)
            samples, attn_mask = cls._retokenize(tokenizer, context_size, text_samples, device)
            batch_size = min(samples.size(0), batch_size)
            n_batches = samples.size(0) // batch_size
            for i in range(n_batches):
                _samples = torch.split(
                    samples[i * batch_size : (i + 1) * batch_size],
                    context_size,
                    dim=-1,
                )
                _attn_mask = torch.split(
                    attn_mask[i * batch_size : (i + 1) * batch_size],
                    context_size,
                    dim=-1,
                )
                for sample_chunk, attn_mask_chunk in zip(_samples, _attn_mask):
                    logits = model(sample_chunk, attention_mask=attn_mask_chunk).logits
                    logits = logits.transpose(-1, -2)
                    nlls = F.cross_entropy(
                        logits[..., :-1],
                        sample_chunk[..., 1:],
                        reduction="none",
                    )
                    first_eos = (sample_chunk == tokenizer.eos_token_id).cumsum(-1) == 1
                    token_mask = sample_chunk != tokenizer.eos_token_id
                    valid_tokens = first_eos[..., 1:] + token_mask[..., 1:]
                    ppl = (nlls * valid_tokens).sum()
                    effective_size += valid_tokens.sum().item()
                    ppls += [ppl.float().cpu().numpy()]
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return float(np.exp(sum(ppls) / effective_size))

    @classmethod
    def compute_mauve(
        cls,
        text_samples: list[str],
        reference_samples: list[str],
        max_text_length: int = 256,
        featurize_model_name: str = DEFAULT_MAUVE_MODEL,
        device: str = "cuda",
        scaling_factor: float = 5.0,
        seed: int = 12345,
        batch_size: int = 16,
        verbose: bool = False,
    ) -> dict[str, float]:
        """MAUVE between generated and human text under a shared feature model.

        Returns the MAUVE score and the frontier integral it is derived from.
        ``p`` is the human reference and ``q`` the model, matching the
        convention of the original paper.
        """
        import mauve as mauve_lib

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # mauve-text addresses the GPU by ordinal, with -1 meaning CPU.
        if device.startswith("cuda") and torch.cuda.is_available():
            _, _, index = device.partition(":")
            device_id = int(index) if index else torch.cuda.current_device()
        else:
            device_id = -1
        result = mauve_lib.compute_mauve(
            p_text=list(reference_samples),
            q_text=list(text_samples),
            featurize_model_name=featurize_model_name,
            max_text_length=max_text_length,
            device_id=device_id,
            mauve_scaling_factor=scaling_factor,
            seed=seed,
            # mauve-text featurizes one text at a time by default, which dominates
            # the runtime of a validation-time call.
            batch_size=batch_size,
            verbose=verbose,
        )
        if device_id >= 0:
            torch.cuda.empty_cache()
        return {
            "mauve": float(result.mauve),
            "mauve_frontier_integral": float(result.frontier_integral),
        }

    @classmethod
    def compute_all(
        cls,
        text_samples: list[str],
        reference_samples: list[str],
        token_batches: list[Tensor],
        *,
        context_size: int,
        ppl_model: str = DEFAULT_PPL_MODEL,
        ppl_batch_size: int = 32,
        mauve_model: str = DEFAULT_MAUVE_MODEL,
        mauve_scaling_factor: float = 5.0,
        mauve_seed: int = 12345,
        mauve_batch_size: int = 16,
        device: str = "cuda",
    ) -> dict[str, float]:
        """Entropy, Gen-PPL, and MAUVE in one pass.

        The two judge models are loaded and released one after another rather
        than together: during training this runs while the trained model, its EMA
        copy, and the optimizer state are all still resident.
        """
        metrics: dict[str, Any] = {
            "generated_sample_count": float(len(text_samples)),
            "reference_sample_count": float(len(reference_samples)),
        }
        if token_batches:
            # Named apart from evaluate.py's corpus-level entropy: this one is the
            # CFM-parity per-batch mean, not a statistic over the pooled samples.
            metrics["gen_entropy"] = cls.compute_mean_entropy(token_batches)
        metrics["gen_ppl"] = cls.compute_mean_gen_ppl(
            text_samples,
            ppl_batch_size,
            context_size=context_size,
            ppl_model=ppl_model,
            device=device,
        )
        metrics["gen_ppl_model"] = ppl_model
        metrics.update(
            cls.compute_mauve(
                text_samples,
                reference_samples,
                max_text_length=context_size,
                featurize_model_name=mauve_model,
                device=device,
                scaling_factor=mauve_scaling_factor,
                seed=mauve_seed,
                batch_size=mauve_batch_size,
            )
        )
        metrics["mauve_model"] = mauve_model
        return metrics
