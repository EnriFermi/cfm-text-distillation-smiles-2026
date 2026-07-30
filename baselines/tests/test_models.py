from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from bcfm_baselines.checkpoint import load_model_state
from bcfm_baselines.config import load_config
from bcfm_baselines.factory import create_model
from bcfm_baselines.models.generative_text import (
    GenerationConfig,
    GenerativeTextModel,
    filter_logits,
)


CONFIGS = [
    "configs/smoke/text8_ar.yaml",
    "configs/smoke/text8_mdlm.yaml",
    "configs/smoke/text8_bd3lm.yaml",
]


class UniformLogitBackbone(torch.nn.Module):
    def __init__(self, vocabulary_size: int = 28) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocabulary_size = vocabulary_size

    def forward(self, input_ids, **kwargs):
        del kwargs
        shape = (*input_ids.shape, self.vocabulary_size)
        return self.anchor + torch.zeros(shape, device=input_ids.device), None


@pytest.mark.parametrize(
    "path,block_size",
    [
        ("configs/text8/bd3lm_b4.yaml", 4),
        ("configs/text8/bd3lm_b8.yaml", 8),
        ("configs/text8/bd3lm_b16.yaml", 16),
        ("configs/text8/bd3lm_b32.yaml", 32),
    ],
)
def test_research_bd3_configs_use_paper_style_initialization_and_sampler(
    path: str, block_size: int
) -> None:
    config = load_config(path)
    assert config["model"]["block_size"] == block_size
    assert config["training"]["require_pretrained"] is True
    assert config["sampling"]["first_hitting"] is True
    assert config["sampling"]["steps_per_block"] == 5000
    assert config["sampling"]["top_p"] == pytest.approx(0.9)
    assert config["sampling"]["top_p_include_boundary"] is False


def test_research_ar_and_mdlm_sampling_matches_official_scripts() -> None:
    ar = load_config("configs/text8/ar.yaml")["sampling"]
    mdlm = load_config("configs/text8/mdlm.yaml")["sampling"]
    assert ar["top_p"] == pytest.approx(0.9)
    assert ar["top_p_include_boundary"] is False
    assert mdlm["num_steps"] == 5000
    assert mdlm["top_p"] == pytest.approx(0.9)
    assert mdlm["top_p_include_boundary"] is False
    assert mdlm["sampling_epsilon"] == pytest.approx(1e-5)
    assert mdlm["noise_removal"] is True
    assert mdlm["cache_denoiser_outputs"] is True


@pytest.mark.parametrize("path", CONFIGS)
def test_shared_interface_backward_and_deterministic_generation(path: str) -> None:
    torch.manual_seed(11)
    config = load_config(path)
    model = create_model(config)
    assert isinstance(model, GenerativeTextModel)
    batch = torch.randint(0, 27, (2, 8))
    metrics = model.compute_loss(batch)
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    model.eval()
    generation = GenerationConfig(**config["sampling"])
    first = model.generate(generation)
    second = model.generate(generation)
    assert torch.equal(first.tokens, second.tokens)
    assert first.tokens.shape == (2, 8)
    assert first.tokens.min() >= 0 and first.tokens.max() < 27
    assert first.diagnostics["num_function_evaluations"] > 0


def test_ar_kv_cache_matches_uncached() -> None:
    model = create_model(load_config("configs/smoke/text8_ar.yaml")).eval()
    base = load_config("configs/smoke/text8_ar.yaml")["sampling"]
    cached = model.generate(GenerationConfig(**{**base, "use_kv_cache": True}))
    uncached = model.generate(GenerationConfig(**{**base, "use_kv_cache": False}))
    assert torch.equal(cached.tokens, uncached.tokens)


def test_ar_scores_every_token_from_internal_bos() -> None:
    config = load_config("configs/smoke/text8_ar.yaml")
    model = create_model(config).eval()
    batch = torch.randint(0, 27, (2, 8))
    result = model.compute_loss(batch)
    bos = torch.full((2, 1), 27, dtype=torch.long)
    model_input = torch.cat((bos, batch[:, :-1]), dim=1)
    logits, _ = model.backbone(model_input, causal=True)
    expected = F.cross_entropy(logits[..., :27].transpose(1, 2), batch)
    assert result["token_count"].item() == batch.numel()
    assert torch.allclose(result["loss"], expected)
    generation = model.generate(GenerationConfig(**config["sampling"]))
    assert config["sampling"]["start_token_id"] == 27
    assert generation.tokens.shape[1] == config["sampling"]["length"]
    assert generation.diagnostics["num_function_evaluations"] == config["sampling"]["length"]
    with pytest.raises(ValueError, match="trained AR BOS"):
        model.generate(GenerationConfig(**{**config["sampling"], "start_token_id": 0}))


def test_bd3_kv_cache_matches_uncached() -> None:
    model = create_model(load_config("configs/smoke/text8_bd3lm.yaml")).eval()
    base = load_config("configs/smoke/text8_bd3lm.yaml")["sampling"]
    cached = model.generate(GenerationConfig(**{**base, "use_kv_cache": True}))
    uncached = model.generate(GenerationConfig(**{**base, "use_kv_cache": False}))
    assert torch.equal(cached.tokens, uncached.tokens)


def test_mdlm_loss_only_uses_masked_positions() -> None:
    model = create_model(load_config("configs/smoke/text8_mdlm.yaml"))
    clean = torch.randint(0, 27, (2, 8))
    time_values = torch.tensor([0.5, 0.5])
    corruption = torch.ones_like(clean, dtype=torch.float)
    corruption[:, :2] = 0.0
    result = model.compute_loss(
        clean, time_values=time_values, corruption_uniform=corruption
    )
    assert result["masked_fraction"].item() == pytest.approx(0.25)
    assert torch.isfinite(result["loss"])


def test_mdlm_objective_has_official_mask_indicator_over_time_weight() -> None:
    model = create_model(load_config("configs/smoke/text8_mdlm.yaml"))
    model.backbone = UniformLogitBackbone()
    clean = torch.randint(0, 27, (2, 8))
    corruption = torch.ones_like(clean, dtype=torch.float)
    corruption[:, :2] = 0.0
    result = model.compute_loss(
        clean,
        time_values=torch.full((2,), 0.5),
        corruption_uniform=corruption,
    )
    expected = 0.5 * torch.log(torch.tensor(27.0))
    assert result["loss"].item() == pytest.approx(expected.item())
    assert result["bits_per_character_estimate"].item() == pytest.approx(
        (expected / torch.log(torch.tensor(2.0))).item()
    )


def test_mdlm_stratified_times_match_official_estimator() -> None:
    model = create_model(load_config("configs/smoke/text8_mdlm.yaml"))
    torch.manual_seed(31)
    actual = model._sample_times(4, torch.device("cpu"))
    torch.manual_seed(31)
    unit = torch.rand(4) / 4 + torch.arange(4) / 4
    expected = model.min_time + (1 - model.min_time) * unit
    assert torch.equal(actual, expected)
    assert all(
        index / 4 <= float(value) < (index + 1) / 4 + model.min_time
        for index, value in enumerate(actual)
    )


def test_mdlm_default_has_no_explicit_time_conditioning() -> None:
    model = create_model(load_config("configs/smoke/text8_mdlm.yaml")).eval()
    tokens = torch.full((2, 8), 27, dtype=torch.long)
    early = model._denoise_logits(tokens, 0.1)
    late = model._denoise_logits(tokens, 0.9)
    assert model.time_conditioning is False
    assert torch.equal(early, late)


def test_mdlm_official_epsilon_cleanup_and_output_cache() -> None:
    config = load_config("configs/smoke/text8_mdlm.yaml")
    model = create_model(config).eval()
    sampling = {
        **config["sampling"],
        "sampling_epsilon": 0.5,
        "num_steps": 4,
        "noise_removal": True,
    }
    cached = model.generate(
        GenerationConfig(**{**sampling, "cache_denoiser_outputs": True})
    )
    uncached = model.generate(
        GenerationConfig(**{**sampling, "cache_denoiser_outputs": False})
    )
    assert torch.equal(cached.tokens, uncached.tokens)
    assert not cached.tokens.eq(27).any()
    assert uncached.diagnostics["num_function_evaluations"] == 5
    assert cached.diagnostics["num_function_evaluations"] <= 5
    assert cached.diagnostics["masked_tokens"][-1] == 0


def test_bd3_one_step_is_valid_exact_transition() -> None:
    config = load_config("configs/smoke/text8_bd3lm.yaml")
    config["sampling"]["steps_per_block"] = 1
    model = create_model(config).eval()
    result = model.generate(GenerationConfig(**config["sampling"]))
    assert result.diagnostics["one_step_is_exact_final_transition"] is True
    assert not result.tokens.eq(27).any()


def test_bd3_first_hitting_reveals_exactly_one_token_per_sequence() -> None:
    config = load_config("configs/smoke/text8_bd3lm.yaml")
    sampling = {
        **config["sampling"],
        "first_hitting": True,
        "steps_per_block": 5000,
        "top_p": 0.9,
    }
    model = create_model(config).eval()
    cached = model.generate(GenerationConfig(**{**sampling, "use_kv_cache": True}))
    uncached = model.generate(GenerationConfig(**{**sampling, "use_kv_cache": False}))
    assert torch.equal(cached.tokens, uncached.tokens)
    assert cached.diagnostics["sampler"] == "first_hitting"
    assert cached.diagnostics["num_function_evaluations"] == 8
    assert cached.diagnostics["masked_tokens"] == [[8, 6, 4, 2, 0]] * 2
    assert all(len(times) == 4 for times in cached.diagnostics["first_hitting_times"])
    assert cached.diagnostics["one_step_is_exact_final_transition"] is False


def test_bd3_nucleus_filter_matches_official_boundary_rule() -> None:
    logits = torch.tensor([0.6, 0.3, 0.1]).log()
    standard = filter_logits(logits, None, 0.8)
    official_bd3 = filter_logits(
        logits, None, 0.8, include_top_p_boundary=False
    )
    assert torch.isfinite(standard).tolist() == [True, True, False]
    assert torch.isfinite(official_bd3).tolist() == [True, False, False]


def test_bd3_clipped_schedule_and_mask_fraction_resampling() -> None:
    model = create_model(load_config("configs/smoke/text8_bd3lm.yaml"))
    assert (model.min_time, 1.0) in model.schedule_candidates()
    assert (model.min_time, 0.5) in model.schedule_candidates()
    assert (0.5, 1.0) in model.schedule_candidates()
    clean = torch.randint(0, 27, (2, 8))
    valid = torch.ones_like(clean, dtype=torch.bool)
    token_t = torch.full_like(clean, 0.75, dtype=torch.float)
    torch.manual_seed(9)
    masked = model._sample_mask(clean, valid, token_t, (0.5, 1.0), None)
    fractions = masked.reshape(2, 2, 4).float().mean(-1)
    assert (fractions >= 0.5).all() and (fractions <= 1.0).all()
    # The official implementation treats epsilon as "no lower clipping".
    no_masks = model._sample_mask(
        clean, valid, torch.zeros_like(token_t), (model.min_time, 0.5), None
    )
    assert not no_masks.any()
    result = model.compute_loss(clean)
    assert result["per_block_nelbo"].shape == (2, 2)


def test_bd3_buffer_bounds_keep_epsilon_and_one_unclipped() -> None:
    # Training reads the bounds back from float32 buffers, where 1e-3 becomes
    # 1.0000000474974513e-3.  Reading that as a clipped lower bound demands a
    # masked token in every block and makes epsilon-rate blocks unsatisfiable.
    model = create_model(load_config("configs/smoke/text8_bd3lm.yaml"))
    bounds = (float(model.mask_rate_min.item()), float(model.mask_rate_max.item()))
    assert bounds[0] > model.min_time
    clean = torch.randint(0, 27, (512, 8))
    valid = torch.ones_like(clean, dtype=torch.bool)
    token_t = torch.full_like(clean, model.min_time, dtype=torch.float)
    torch.manual_seed(11)
    # Reading the buffer bound as clipped raises instead of returning, and the
    # blocks it does return are all forced up to one masked token.
    masked = model._sample_mask(clean, valid, token_t, bounds, None)
    empty_blocks = ~masked.reshape(512, 2, model.block_size).any(-1)
    assert empty_blocks.float().mean() > 0.9


def test_bd3_clipped_bounds_never_exhaust_the_resampler() -> None:
    model = create_model(load_config("configs/smoke/text8_bd3lm.yaml"))
    clean = torch.randint(0, 27, (256, 8))
    valid = torch.ones_like(clean, dtype=torch.bool)
    torch.manual_seed(13)
    for lower, upper in model.schedule_candidates():
        model.set_mask_rate_bounds(lower, upper)
        bounds = (float(model.mask_rate_min.item()), float(model.mask_rate_max.item()))
        block_t = model._sample_block_times(256, 2, clean.device, bounds)
        token_t = block_t.repeat_interleave(model.block_size, dim=1)
        masked = model._sample_mask(clean, valid, token_t, bounds, None)
        fractions = masked.reshape(256, 2, model.block_size).float().mean(-1)
        if lower > model.min_time:
            assert (fractions >= lower - 1.0 / model.block_size).all()
        if upper < 1.0:
            assert (fractions <= upper).all()


def test_bd3_objective_has_per_block_official_time_weight() -> None:
    model = create_model(load_config("configs/smoke/text8_bd3lm.yaml"))
    model.backbone = UniformLogitBackbone()
    clean = torch.randint(0, 27, (2, 8))
    corruption = torch.ones_like(clean, dtype=torch.float)
    corruption[:, 0:2] = 0.0
    corruption[:, 4:6] = 0.0
    result = model.compute_loss(
        clean,
        time_values=torch.full((2, 2), 0.5),
        corruption_uniform=corruption,
    )
    expected = torch.log(torch.tensor(27.0))
    assert result["loss"].item() == pytest.approx(expected.item())
    assert torch.allclose(result["per_block_nelbo"], expected.expand(2, 2))


def test_legacy_bd3_state_without_schedule_buffers_still_loads() -> None:
    config = load_config("configs/smoke/text8_bd3lm.yaml")
    source = create_model(config)
    legacy_state = {
        name: value
        for name, value in source.state_dict().items()
        if name not in {"mask_rate_min", "mask_rate_max"}
    }
    restored = create_model(config)
    load_model_state(restored, legacy_state)
    assert restored.mask_rate_min.item() == pytest.approx(config["model"]["mask_rate_min"])
    assert restored.mask_rate_max.item() == pytest.approx(config["model"]["mask_rate_max"])
