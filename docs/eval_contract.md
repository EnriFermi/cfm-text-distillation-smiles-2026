# Eval contract: sampling and scoring are separate processes

Adding a metric must never require re-running a model. So evaluation is split in
two, with a file as the only interface between them.

```
┌────────────────────┐   dumps/*.json    ┌────────────────────┐
│ eval/dump_samples  │ ────────────────► │ eval/score_texts   │
│ model → texts      │  + *.tokens.npz   │ texts → metrics    │
│ (only stage that   │                   │ (never loads the   │
│  loads a ckpt)     │                   │  model under test) │
└────────────────────┘                   └────────────────────┘
```

Real data goes through the same pipe (`--gold`), so the reference line is scored
by exactly the same code as the models.

## Process 1 — `eval/dump_samples.py`

Takes a checkpoint, samples texts, writes a dump. Architecture is **auto-detected**
from tensor shapes (vocab/hidden/layers/heads/cond_dim/length/embedding type) and
`block_size` from the saved hyper-parameters, so one command covers every model
in the repo — text8 or TinyStories, full-sequence or block.

```sh
# block model (M3): sampler auto-resolves to block_causal
python -m eval.dump_samples --checkpoint logs/train/<run>/checkpoints/step_0070000.ckpt \
    --nfe 1 2 4 8 16 32 --discretize argmax sample --n-samples 512 \
    --out dumps/ts_m3_step70000.json

# full-sequence model (M1)
python -m eval.dump_samples --checkpoint baseline/s_baseline.ckpt \
    --nfe 1 2 4 8 --out dumps/text8_m1_s_baseline.json

# same full-sequence model wrapped blockwise (M2, training-free)
python -m eval.dump_samples --checkpoint baseline/s_baseline.ckpt \
    --sampler blockwise_infer --infer-block-size 16 \
    --nfe 1 2 4 --out dumps/text8_m2_s_baseline_b16.json

# real data, identical schema
python -m eval.dump_samples --gold tinystories --n-samples 2048 \
    --out dumps/gold_tinystories.json
```

Samplers: `full_cfm` (`sample_flow_map_batch`), `block_causal` (M3,
`block/sampling.py`), `blockwise_infer` (M2, wraps a full-sequence model),
`gold`. `auto` picks block vs full from the checkpoint.

## Dump schema (`bcfm-textdump/v1`)

`<name>.json`

```json
{ "schema": "bcfm-textdump/v1", "label": "...", "created_utc": "...",
  "git_commit": "...",
  "model":   { "kind": "checkpoint|gold", "checkpoint": "...",
               "checkpoint_step": 70000, "checkpoint_sha256_head": "...",
               "dataset": "tinystories", "tokenizer": "gpt2",
               "arch": { "vocab_size": 50257, "hidden_size": 384, ... } },
  "sampler": { "kind": "block_causal", "block_size": 16, "length": 256, "seed": 0 },
  "points":  [ { "id": "argmax_nfe4", "nfe": 4, "discretize": "argmax",
                 "forwards_per_sequence": 64, "n_samples": 512,
                 "sampling_seconds": 51.0, "texts": ["...", "..."] } ] }
```

`<name>.tokens.npz` holds the token ids per point, so entropy and n-gram stats
are computed on the ids the model actually emitted rather than on a
re-tokenization. Dumps are written incrementally after every point, so an
interrupted run still yields usable data.

## Process 2 — `eval/score_texts.py`

Input is dumps only. It may load an *external* model (a judge LM, GPT-2 for the
MAUVE featurizer) but never the model under test.

```sh
python -m eval.score_texts --dumps 'dumps/ts_*.json' \
    --reference dumps/gold_tinystories.json \
    --metrics entropy rep diversity zipf ngram mauve \
    --out results/ts_scores.json
```

| group | what it measures | needs reference |
|---|---|---|
| `entropy` | mean within-sequence token entropy (nats) | no |
| `rep` | `seq_rep_{2,3,4}`, `distinct_{2,3,4}` — degenerate repetition | no |
| `diversity` | cross-sample 4-gram Jaccard overlap — mode collapse | no |
| `zipf` | Zipf slope; natural text sits near 1 | no |
| `ngram` | Jensen-Shannon divergence of uni/bigram stats, in nats | yes |
| `mauve` | distribution distance in GPT-2 embedding space | yes |
| `gen_ppl` | perplexity under `--judge`, tokenized by that judge | no |

`gen_ppl` deliberately does **not** reuse
`semicat/metric/text_dist.py::compute_mean_gen_ppl`: that function hardcodes the
gpt2-large tokenizer, which is correct only for judges sharing the GPT-2 vocab
(GPT-J). Everything else about it — truncation, mask-after-first-eos,
`PPL = exp(sum NLL / valid tokens)` — is reproduced exactly, so numbers stay
comparable within a judge.

## Sanity checks

Scoring the gold dump against itself gives `js_1gram = js_2gram = 0.0000` and
`mauve = 1.0000`. MAUVE of real text against degenerate repetition is `0.004`,
against real text `0.97` — the scale to read every score on.

## Caveat that motivated all this

On TinyStories, `gen_ppl` and `mauve` rank NFE settings **oppositely** (PPL best
at NFE=4 where the text degenerates; MAUVE best at NFE=16 where entropy matches
the data). A stronger judge does not fix it. Report MAUVE plus entropy; treat
gen-PPL as an illustration of the failure mode, not as the quality metric.
