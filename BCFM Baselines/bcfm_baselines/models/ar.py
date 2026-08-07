"""Decoder-only autoregressive Text8 baseline."""

from __future__ import annotations

import time

import torch
from torch import Tensor
from torch.nn import functional as F

from bcfm_baselines.models.generative_text import (
    GenerationConfig,
    GenerationResult,
    GenerativeTextModel,
    batch_tokens,
    sample_logits,
    seeded_generator,
)
from bcfm_baselines.net.text_transformer import SharedTextTransformer, TransformerConfig


class AutoregressiveTextModel(GenerativeTextModel):
    model_type = "ar"

    def __init__(
        self,
        backbone: TransformerConfig | dict,
        data_vocab_size: int = 27,
        mask_token_id: int = 27,
        bos_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = SharedTextTransformer(backbone)
        self.data_vocab_size = data_vocab_size
        self.mask_token_id = mask_token_id
        self.bos_token_id = mask_token_id if bos_token_id is None else bos_token_id
        if not data_vocab_size <= self.bos_token_id < self.backbone.config.vocab_size:
            raise ValueError("bos_token_id must be a model-only vocabulary ID")

    def _data_logits(self, logits: Tensor) -> Tensor:
        return logits[..., : self.data_vocab_size]

    def compute_loss(self, batch: Tensor | dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        del kwargs
        tokens, valid = batch_tokens(batch)
        bos = torch.full(
            (tokens.shape[0], 1),
            self.bos_token_id,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        inputs = torch.cat((bos, tokens[:, :-1]), dim=1)
        targets = tokens
        target_valid = valid
        logits, _ = self.backbone(inputs, causal=True)
        token_nll = F.cross_entropy(
            self._data_logits(logits).transpose(1, 2), targets, reduction="none"
        )
        loss = (token_nll * target_valid).sum() / target_valid.sum().clamp_min(1)
        return {
            "loss": loss,
            "nll": loss.detach(),
            "bits_per_character": (loss.detach() / torch.log(loss.new_tensor(2.0))),
            "token_count": target_valid.sum().detach(),
        }

    @torch.no_grad()
    def generate(
        self,
        generation_config: GenerationConfig,
        prompt: Tensor | None = None,
        **kwargs,
    ) -> GenerationResult:
        del kwargs
        config = generation_config
        device = next(self.parameters()).device
        generator = seeded_generator(device, config.seed)
        if config.start_token_id != self.bos_token_id:
            raise ValueError(
                f"start_token_id must match the trained AR BOS ID {self.bos_token_id}"
            )
        if prompt is None:
            generated = torch.empty(
                (config.batch_size, 0), device=device, dtype=torch.long
            )
        else:
            generated = prompt.to(device=device, dtype=torch.long)
            if generated.shape[0] != config.batch_size:
                raise ValueError("prompt batch size must match generation config")
            if generated.numel() and (
                int(generated.min().item()) < 0
                or int(generated.max().item()) >= self.data_vocab_size
            ):
                raise ValueError("prompt contains an ID outside the data vocabulary")
        if generated.shape[1] > config.length:
            raise ValueError("prompt is longer than requested output length")
        start = torch.full(
            (config.batch_size, 1),
            config.start_token_id,
            device=device,
            dtype=torch.long,
        )
        context = torch.cat((start, generated), dim=1)
        cache = None
        started = time.perf_counter()
        nfe = 0
        while generated.shape[1] < config.length:
            if config.use_kv_cache:
                model_input = context if cache is None else context[:, -1:]
                logits, cache = self.backbone(
                    model_input, causal=True, cache=cache, return_cache=True
                )
            else:
                logits, _ = self.backbone(context, causal=True)
            next_token, _ = sample_logits(
                self._data_logits(logits[:, -1]), config, generator
            )
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            context = torch.cat((context, next_token[:, None]), dim=1)
            nfe += 1
        return GenerationResult(
            tokens=generated,
            diagnostics={
                "num_function_evaluations": nfe,
                "wall_clock_seconds": time.perf_counter() - started,
                "kv_cache": config.use_kv_cache,
                "seed": config.seed,
            },
        )
