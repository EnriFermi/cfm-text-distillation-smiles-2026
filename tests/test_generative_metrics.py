"""Gen-PPL / MAUVE wiring, exercised without downloading the judge models.

The judges themselves are `transformers`/`mauve-text` and are covered by their own
projects; what can silently break here is the plumbing -- how many samples get
drawn, whether each chunk gets a distinct seed, whether the reference set comes
from the right split, and whether the results reach the validation record. Those
are what this file pins down, with the two expensive calls stubbed out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from bcfm_baselines.config import load_config
from bcfm_baselines.data import create_corpus
from bcfm_baselines.generative_eval import (
    GenerativeEvalConfig,
    evaluate_generation,
    reference_texts,
    require_metric_dependencies,
    resolve_config,
    sample_texts,
)
from bcfm_baselines.metrics import TextMetrics
from bcfm_baselines.models.generative_text import GenerationResult
from bcfm_baselines.train import train


class _RecordingModel:
    """Minimal stand-in for a baseline: records the seeds it is generated with."""

    model_type = "mdlm"

    def __init__(self, length: int = 8, vocab_size: int = 27) -> None:
        self.length = length
        self.vocab_size = vocab_size
        self.seeds: list[int] = []
        self.batch_sizes: list[int] = []
        self.steps: list[int] = []
        self.steps_per_block: list[int | None] = []
        self.first_hitting: list[bool] = []
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def generate(self, generation_config) -> GenerationResult:
        self.seeds.append(generation_config.seed)
        self.batch_sizes.append(generation_config.batch_size)
        self.steps.append(generation_config.num_steps)
        self.steps_per_block.append(generation_config.steps_per_block)
        self.first_hitting.append(generation_config.first_hitting)
        generator = torch.Generator().manual_seed(generation_config.seed)
        tokens = torch.randint(
            0,
            self.vocab_size,
            (generation_config.batch_size, generation_config.length),
            generator=generator,
        )
        return GenerationResult(tokens=tokens, diagnostics={"seed": generation_config.seed})


@pytest.fixture
def stub_judges(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the two model-loading metrics with deterministic stand-ins."""
    seen: dict = {}

    def fake_gen_ppl(text_samples, batch_size, context_size=1024, ppl_model="", device="cpu"):
        seen["gen_ppl"] = {
            "samples": len(text_samples),
            "batch_size": batch_size,
            "context_size": context_size,
            "model": ppl_model,
        }
        return 12.5

    def fake_mauve(text_samples, reference_samples, **kwargs):
        seen["mauve"] = {
            "samples": len(text_samples),
            "references": len(reference_samples),
            "max_text_length": kwargs.get("max_text_length"),
            "model": kwargs.get("featurize_model_name"),
        }
        return {"mauve": 0.75, "mauve_frontier_integral": 0.25}

    monkeypatch.setattr(TextMetrics, "compute_mean_gen_ppl", staticmethod(fake_gen_ppl))
    monkeypatch.setattr(TextMetrics, "compute_mauve", staticmethod(fake_mauve))
    return seen


def test_entropy_matches_the_closed_form() -> None:
    # Two ids, 3:1 -- entropy = -(0.75 ln 0.75 + 0.25 ln 0.25).
    batch = torch.tensor([[0, 0, 0, 1]])
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert TextMetrics.compute_mean_entropy([batch]) == pytest.approx(expected)
    # A uniform batch of one distinct id carries no information.
    assert TextMetrics.compute_mean_entropy([torch.zeros(4, dtype=torch.long)]) == 0.0


def test_resolve_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown generative_evaluation keys"):
        resolve_config({"generative_evaluation": {"sample_count": 4, "typo": 1}})


def test_resolve_config_defaults_to_the_cfm_judges() -> None:
    settings = resolve_config({})
    assert settings.ppl_model == "gpt2-large"
    assert settings.mauve_model == "gpt2-large"


def test_sampling_is_chunked_with_distinct_seeds(text8_smoke_dir: Path) -> None:
    corpus = create_corpus(
        {"dataset": {"name": "text8", "data_dir": str(text8_smoke_dir), "sequence_length": 8,
                     "reference_split": False}}
    )
    model = _RecordingModel(length=8, vocab_size=corpus.vocab_size)
    settings = GenerativeEvalConfig(
        sample_count=5, sampling_batch_size=2, seed=100, num_steps=3
    )
    texts, batches, diagnostics = sample_texts(
        model, corpus, {"length": 8, "num_steps": 64, "steps_per_block": None}, settings
    )
    assert len(texts) == 5
    # A trailing chunk smaller than the batch is still drawn, not dropped.
    assert model.batch_sizes == [2, 2, 1]
    # Distinct seeds: one seed for every chunk would repeat the same samples.
    assert model.seeds == [100, 101, 102]
    # The validation step budget overrides the config's full sampling budget.
    assert model.steps == [3, 3, 3]
    assert sum(batch.shape[0] for batch in batches) == 5
    assert diagnostics["generation_chunks"] == 3


def test_steps_per_block_is_only_applied_to_block_models(text8_smoke_dir: Path) -> None:
    corpus = create_corpus(
        {"dataset": {"name": "text8", "data_dir": str(text8_smoke_dir), "sequence_length": 8,
                     "reference_split": False}}
    )
    model = _RecordingModel(length=8, vocab_size=corpus.vocab_size)
    settings = GenerativeEvalConfig(sample_count=1, sampling_batch_size=1, steps_per_block=2)
    # AR and MDLM carry steps_per_block=None; overriding it would build an invalid
    # block schedule for a model that has no blocks.
    sample_texts(model, corpus, {"length": 8, "steps_per_block": None}, settings)
    sample_texts(model, corpus, {"length": 8, "steps_per_block": 16}, settings)


def test_validation_can_disable_first_hitting_without_changing_offline_sampling(
    text8_smoke_dir: Path,
) -> None:
    corpus = create_corpus(
        {"dataset": {"name": "text8", "data_dir": str(text8_smoke_dir), "sequence_length": 8,
                     "reference_split": False}}
    )
    model = _RecordingModel(length=8, vocab_size=corpus.vocab_size)
    sampling = {"length": 8, "steps_per_block": 5000, "first_hitting": True}
    validation = GenerativeEvalConfig(
        sample_count=1,
        sampling_batch_size=1,
        steps_per_block=2,
        first_hitting=False,
    )
    sample_texts(model, corpus, sampling, validation)
    offline = GenerativeEvalConfig(
        sample_count=1,
        sampling_batch_size=1,
        steps_per_block=None,
        first_hitting=None,
    )
    sample_texts(model, corpus, sampling, offline)
    assert model.steps_per_block == [2, 5000]
    assert model.first_hitting == [False, True]


def test_reference_texts_come_from_the_named_split(text8_smoke_dir: Path) -> None:
    corpus = create_corpus(
        {"dataset": {"name": "text8", "data_dir": str(text8_smoke_dir), "sequence_length": 8,
                     "reference_split": False}}
    )
    texts = reference_texts(corpus, 6, seed=3, split="validation")
    assert len(texts) == 6
    # Deterministic: the human reference must not drift between validations.
    assert texts == reference_texts(corpus, 6, seed=3, split="validation")
    assert texts != reference_texts(corpus, 6, seed=4, split="validation")


def test_evaluate_generation_reports_both_metrics(text8_smoke_dir: Path, stub_judges) -> None:
    corpus = create_corpus(
        {"dataset": {"name": "text8", "data_dir": str(text8_smoke_dir), "sequence_length": 8,
                     "reference_split": False}}
    )
    model = _RecordingModel(length=8, vocab_size=corpus.vocab_size)
    settings = GenerativeEvalConfig(sample_count=4, sampling_batch_size=2, ppl_batch_size=2)
    values = evaluate_generation(model, corpus, {"length": 8}, settings, "cpu")

    assert values["gen_ppl"] == 12.5
    assert values["mauve"] == 0.75
    assert values["mauve_frontier_integral"] == 0.25
    assert values["gen_entropy"] > 0
    assert values["generated_sample_count"] == 4
    assert values["reference_sample_count"] == 4
    # The judges see the corpus sequence length as their context, not 1024.
    assert stub_judges["gen_ppl"]["context_size"] == 8
    assert stub_judges["mauve"]["max_text_length"] == 8
    assert stub_judges["gen_ppl"]["model"] == "gpt2-large"
    # The model is handed back in the mode it arrived in.
    assert model.training


def test_disabled_generative_evaluation_is_a_no_op(text8_smoke_dir: Path) -> None:
    corpus = create_corpus(
        {"dataset": {"name": "text8", "data_dir": str(text8_smoke_dir), "sequence_length": 8,
                     "reference_split": False}}
    )
    model = _RecordingModel(length=8, vocab_size=corpus.vocab_size)
    settings = GenerativeEvalConfig(enabled=False)
    assert evaluate_generation(model, corpus, {"length": 8}, settings, "cpu") == {}
    assert model.seeds == []


def test_metric_dependencies_fail_before_expensive_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bcfm_baselines.generative_eval.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name))
        if name == "transformers"
        else object(),
    )
    with pytest.raises(RuntimeError, match="transformers==4.41.0"):
        require_metric_dependencies(GenerativeEvalConfig(enabled=True))
    require_metric_dependencies(GenerativeEvalConfig(enabled=False))


def test_validation_record_carries_gen_ppl_and_mauve(
    text8_smoke_dir: Path, tmp_path: Path, stub_judges
) -> None:
    """The end-to-end path: a training run writes both metrics into validation.jsonl."""
    config = load_config("configs/smoke/text8_mdlm.yaml")
    config["dataset"]["data_dir"] = str(text8_smoke_dir)
    config["generative_evaluation"].update(enabled=True, sample_count=2, sampling_batch_size=2)
    run_dir = tmp_path / "run"
    train(config, run_dir, resume="none")

    records = [
        json.loads(line)
        for line in (run_dir / "validation.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert records, "training wrote no validation record"
    assert records[-1]["gen_ppl"] == 12.5
    assert records[-1]["mauve"] == 0.75
    assert records[-1]["sample_texts"]
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["generative_evaluation"]["enabled"] is True
    assert (run_dir / "training.jsonl").exists()
    assert (run_dir / "validation_samples" / "step_00000001.json").exists()
