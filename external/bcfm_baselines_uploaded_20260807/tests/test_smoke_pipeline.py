from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from bcfm_baselines.checkpoint import load_checkpoint_model
from bcfm_baselines.config import load_config
from bcfm_baselines.evaluate import evaluate
from bcfm_baselines.models.generative_text import GenerationConfig
from bcfm_baselines.factory import create_model
from bcfm_baselines.train import _load_pretrained_backbone, apply_overrides, train, validate


def test_set_overrides_coerce_types_and_reject_unknown_keys() -> None:
    config = load_config("configs/text8/ar.yaml")
    apply_overrides(
        config,
        [
            "training.batch_size=256",
            "training.gradient_accumulation_steps=1",
            "model.time_conditioning=true",
        ],
    )
    assert config["training"]["batch_size"] == 256
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["model"]["time_conditioning"] is True
    with pytest.raises(KeyError):
        apply_overrides(config, ["training.nope=1"])
    with pytest.raises(ValueError):
        apply_overrides(config, ["training.batch_size"])


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/smoke/text8_ar.yaml",
        "configs/smoke/text8_mdlm.yaml",
        "configs/smoke/text8_bd3lm.yaml",
    ],
)
def test_one_step_checkpoint_load_sample_and_metrics(text8_smoke_dir: Path, tmp_path: Path, config_path: str) -> None:
    config = load_config(config_path)
    config["dataset"]["data_dir"] = str(text8_smoke_dir)
    run_dir = tmp_path / Path(config_path).stem
    train(config, run_dir, resume="none")
    checkpoint = run_dir / "checkpoints" / "last.pt"
    assert checkpoint.exists()
    model, payload = load_checkpoint_model(checkpoint)
    assert payload["step"] == 1
    assert payload["ema_model"] is not None
    assert payload["weights_used_for_inference"] == "ema"
    generation = GenerationConfig(**config["sampling"])
    assert torch.equal(model.generate(generation).tokens, model.generate(generation).tokens)
    results = evaluate(
        Namespace(
            checkpoint=checkpoint,
            device="cpu",
            eval_batch_size=2,
            eval_batches=1,
            num_samples=2,
            seed=7,
            num_steps=2,
            steps_per_block=2 if config["model"]["type"] == "bd3lm" else None,
        )
    )
    assert results["model_type"] == config["model"]["type"]
    assert results["weights_used_for_inference"] == "ema"
    assert results["sampling_tokens_per_second"] > 0
    assert results["sample_vocabulary_coverage"] > 0
    if config["model"]["type"] != "ar":
        assert results["likelihood_metric_type"] == (
            "single_sample_epsilon_truncated_continuous_time_nelbo"
        )
        assert results["likelihood_is_strict_upper_bound"] is False
        assert "test_epsilon_truncated_nelbo_perplexity_estimate" in results
        repeated = evaluate(
            Namespace(
                checkpoint=checkpoint,
                device="cpu",
                eval_batch_size=2,
                eval_batches=1,
                num_samples=2,
                seed=7,
                num_steps=2,
                steps_per_block=2 if config["model"]["type"] == "bd3lm" else None,
            )
        )
        assert repeated["test_nelbo"] == results["test_nelbo"]
        assert repeated["sample_texts"] == results["sample_texts"]


def test_bd3_backbone_initializes_from_mdlm_ema(
    text8_smoke_dir: Path, tmp_path: Path
) -> None:
    mdlm_config = load_config("configs/smoke/text8_mdlm.yaml")
    mdlm = create_model(mdlm_config)
    checkpoint = tmp_path / "mdlm.pt"
    torch.save(
        {
            "model": mdlm.state_dict(),
            "ema_model": mdlm.state_dict(),
            "step": 17,
            "config": mdlm_config,
        },
        checkpoint,
    )
    bd3 = create_model(load_config("configs/smoke/text8_bd3lm.yaml"))
    metadata = _load_pretrained_backbone(bd3, checkpoint, torch.device("cpu"))
    assert metadata["type"] == "mdlm_pretrained_backbone"
    assert metadata["checkpoint_step"] == 17
    assert metadata["weights"] == "ema"
    for name, value in mdlm.backbone.state_dict().items():
        assert torch.equal(value, bd3.backbone.state_dict()[name])

    bd3_config = load_config("configs/smoke/text8_bd3lm.yaml")
    bd3_config["dataset"]["data_dir"] = str(text8_smoke_dir)
    bd3_config["training"]["from_pretrained"] = str(checkpoint)
    run_dir = tmp_path / "pretrained_bd3"
    train(bd3_config, run_dir, resume="none")
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert run_metadata["initialization"]["type"] == "mdlm_pretrained_backbone"
    saved = torch.load(run_dir / "checkpoints" / "last.pt", weights_only=False)
    assert saved["initialization"] == run_metadata["initialization"]


def test_research_bd3_cannot_silently_train_from_scratch(
    text8_smoke_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BD3_PRETRAIN_CHECKPOINT", raising=False)
    config = load_config("configs/smoke/text8_bd3lm.yaml")
    config["dataset"]["data_dir"] = str(text8_smoke_dir)
    config["training"]["require_pretrained"] = True
    with pytest.raises(ValueError, match="requires an MDLM checkpoint"):
        train(config, tmp_path / "must_not_start_from_scratch", resume="none")


def test_resume_advances_optimizer_step(text8_smoke_dir: Path, tmp_path: Path) -> None:
    config = load_config("configs/smoke/text8_ar.yaml")
    config["dataset"]["data_dir"] = str(text8_smoke_dir)
    run_dir = tmp_path / "resume"
    train(config, run_dir, resume="none")
    config["training"]["max_steps"] = 2
    train(config, run_dir, resume="auto")
    payload = torch.load(run_dir / "checkpoints" / "last.pt", weights_only=False)
    assert payload["step"] == 2
    assert payload["processed_tokens"] == 2 * 2 * 8


def test_validation_metrics_are_weighted_by_token_count() -> None:
    class UnevenCorpus:
        def sequential_batches(self, split, batch_size, max_batches):
            del split, batch_size, max_batches
            yield torch.zeros((2, 3), dtype=torch.long)
            yield torch.zeros((1, 3), dtype=torch.long)

    class BatchSizeMetric(torch.nn.Module):
        model_type = "ar"

        def compute_loss(self, batch):
            value = torch.tensor(float(batch.shape[0]))
            return {"loss": value, "nll": value, "token_count": torch.tensor(batch.numel())}

    results = validate(
        BatchSizeMetric(), UnevenCorpus(), torch.device("cpu"), 2, 2, "float32"
    )
    assert results["token_count"] == 9
    assert results["nll"] == pytest.approx((2 * 6 + 1 * 3) / 9)
