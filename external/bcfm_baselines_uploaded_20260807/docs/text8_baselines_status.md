# Text8 baselines Stage 1 status

Updated: 2026-07-22.

## Implementation and local validation

- [x] Deterministic Text8 preparation and metadata/checksums.
- [x] Shared character vocabulary, split mode, sequence length, and backbone.
- [x] Config-selected AR, MDLM, and BD3-LM training.
- [x] Checkpoint save, load, and resume (`last.pt` + `best.pt` by validation
  NLL/NELBO) with TensorBoard scalar logging under `runs/<run>/tensorboard`.
- [x] AR full-token NLL from internal BOS,
  greedy/sample/temperature/top-k/top-p, and exact KV cache.
- [x] MDLM epsilon-grid DDPM-cache generation, final noise removal, and
  requested diagnostics.
- [x] BD3-LM block generation, exact block mask, data-driven clipped schedule,
  masked-fraction resampling, official first-hitting sampling, and
  finalized-prefix KV cache.
- [x] Strict MDLM-to-BD3 EMA-backbone initialization for research configs;
  the supplied research BD3 configs reject scratch training.
- [x] EMA training/inference and compatible loading of legacy BD3-LM weights
  that predate the schedule buffers.
- [x] Token-weighted offline likelihood/BPC, sample statistics, throughput, and
  aggregation.
- [x] Fixed-seed generation and diffusion-likelihood reproducibility.
- [x] CPU smoke suite: `31 passed` on 2026-07-22.
- [x] Local command pipeline completed for all three smoke configs: prepare ->
  train -> checkpoint -> sample -> evaluate -> aggregate.
- [x] Research-size AR, MDLM, and BD3-LM b16 forward/loss completed at context
  256; each model has exactly 92,244,764 trainable parameters.
- [ ] Canonical Text8 prepared on the target cluster.
- [ ] Full AR Slurm training completed.
- [ ] Full MDLM Slurm training completed.
- [ ] Full BD3-LM b4/b8/b16/b32 Slurm training completed.
- [ ] At least one preempted/resubmitted cluster job resumed successfully.
- [ ] All requested diffusion step-count evaluations completed.
- [ ] Multi-seed metrics aggregated and compared with existing CFM results.
- [ ] GPU type, peak memory, throughput, and wall time reviewed.

The Stage 2 TinyStories data layer is now implemented (byte-level tokenizer,
`prepare_tinystories`, `TinyStoriesCorpus`, `configs/tinystories/*`), reusing
the same models/trainer/evaluator. Its cluster runs share the unchecked items
above; the single `slurm/run_all.sbatch` job executes Text8 first, then
TinyStories.

## Algorithm references checked

The final audit on 2026-07-22 used only primary sources:

- MDLM [paper](https://arxiv.org/abs/2406.07524) and official
  [repository at `c112c526`](https://github.com/kuleshov-group/mdlm/tree/c112c526d193436838c98d81455ee51f90309470).
- BD3-LM [paper](https://arxiv.org/abs/2503.09573) and official
  [repository at `1c3e8f43`](https://github.com/kuleshov-group/bd3lms/tree/1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7).

Those hashes were also verified against the repositories' current remote HEADs.

## Exact commands

CPU smoke validation used:

```bash
python -m bcfm_baselines.data.prepare_text8 \
  --output-dir data/text8-smoke --smoke --force
pytest -q
```

The command-level smoke run used each of
`configs/smoke/text8_{ar,mdlm,bd3lm}.yaml` with the same `train`, `sample`, and
`evaluate` module commands shown below for research runs, followed by
`python -m bcfm_baselines.aggregate`. Outputs were written under `/tmp` and are
not research results.

The whole study (both datasets, all three models, evaluation, aggregation) runs
from a single resumable job. Text8 is executed first, then TinyStories:

```bash
sbatch slurm/run_all.sbatch
```

After preemption, resubmit the same command; `--resume auto` loads each
`runs/$RUN_NAME/checkpoints/last.pt`, and stages already at `max_steps` exit
immediately.

Standalone sample and evaluation examples:

```bash
python -m bcfm_baselines.sample \
  --checkpoint runs/mdlm_text8_seed12345/checkpoints/last.pt \
  --output runs/mdlm_text8_seed12345/samples_steps64.json --num-steps 64

python -m bcfm_baselines.evaluate \
  --checkpoint runs/bd3lm_b16_text8_seed12345/checkpoints/last.pt \
  --output runs/bd3lm_b16_text8_seed12345/evaluation_steps16.json \
  --no-first-hitting --steps-per-block 16

python -m bcfm_baselines.aggregate runs/*/evaluation.json --output runs/aggregate.json
```

To sweep MDLM steps, run evaluation with `--num-steps` in
`16 32 64 128 256`. To sweep BD3-LM, use `--steps-per-block` in
`1 2 4 8 16 32` together with `--no-first-hitting`, keeping checkpoint, sample
count, and seed fixed. The research default is first-hitting sampling, which
finishes after exactly one reveal per block position.
