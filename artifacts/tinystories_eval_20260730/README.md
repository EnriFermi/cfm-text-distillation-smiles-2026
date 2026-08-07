# TinyStories / text8 evaluation artifacts (2026-07-30)

Durable copy of the evaluation work done in the scratchpad, which is a temp
directory and has already been wiped once by a machine reboot.

All PPL numbers are generative perplexity under **GPT-J-6B** via the untouched
upstream pipeline (`semicat/metric/text_dist.py::compute_mean_gen_ppl`,
gpt2-large tokenizer, mask-after-eos, `PPL = exp(sum NLL / valid tokens)`),
length 256, seed 0.

## Gold references (the floor every number is read against)

| data | gen-PPL | entropy |
|---|---:|---:|
| real text8 (test, full corpus) | 75.64 | 2.856 |
| real TinyStories (validation) | **6.40** | **5.902** |

Never compare these two across datasets — text8 is a character stream GPT-J
predicts badly, TinyStories is ordinary English. Use the ratio to each dataset's
own gold. Files: `gold_reference_text8.json`, `gold_reference_tinystories.json`.

## Files

| file | what |
|---|---|
| `gen_ppl_grid_ts_step70000.json` | NFE 1-128 x {argmax, sample}, block mode, TS finetune step 70000 |
| `gen_ppl_grid_ts_step70000_nfe256.json` | the NFE=256 points of the same grid |
| `gen_ppl_bcfm_ts_src68001_vs_ft40000.json` | source CFM vs finetune @40k, both in block mode |
| `gen_ppl_bcfm_text8_sweep.json` | text8 M3 checkpoint series (10 ckpts x spb 2/4/8) |
| `gen_text_grid_ts.json/.txt` | sample texts, NFE 1-128 x both discretizers (3 each) |
| `text_nfe4_vs_256.json/.txt` | sample texts at NFE 4 vs 256 |
| `mauve_grid_ts_step70000.json` | MAUVE over the NFE grid (see below) |
| `mauve_grid_texts_step70000.json` | **512 generated texts per grid point** + 2048 reference texts |
| `scripts/` | everything needed to reproduce |

`mauve_grid_texts_step70000.json` is the important one for future work: it holds
the actual generations, so reference-free metrics (rep-n, distinct-n, n-gram
divergence, self-BLEU) can be computed without re-sampling. Because seed,
checkpoint and batch size match, these are the *same* texts that produced the
gen-PPL grid numbers.

## Headline result: NFE x discretizer grid (TS finetune step 70000, B=16, 512 samples)

| NFE | fwd/seq | argmax PPL / ent | sample PPL / ent |
|---:|---:|---:|---:|
| 1 | 16 | 51.87 / 4.365 | 60.46 / 4.469 |
| 2 | 32 | 52.13 / 4.572 | 63.25 / 4.712 |
| 4 | 64 | **46.80** / 4.964 | **53.77** / 5.064 |
| 8 | 128 | 47.86 / 5.508 | 56.53 / 5.599 |
| 16 | 256 | 54.51 / 5.912 | 67.60 / 6.004 |
| 32 | 512 | 58.02 / 6.079 | 71.60 / 6.153 |
| 64 | 1024 | 57.72 / 6.100 | 66.34 / 6.141 |
| 128 | 2048 | 55.43 / 6.067 | 59.98 / 6.092 |
| 256 | 4096 | 52.50 / 6.063 | 52.79 / 6.060 |

- argmax wins 8/9; the gap collapses to +0.29 at NFE=256 because the endpoint
  distribution goes nearly degenerate and both discretizers pick the same token.
- PPL minimum at NFE=4, but **entropy there is 4.96 vs gold 5.902** — the model is
  under-diverse at its best-scoring point. Gold entropy is reached at NFE=16.
- Read together with MAUVE below: gen-PPL rewards the repetition visible at low
  NFE (`toy ... toy time ... toy toys ... always said toy`).

## Source vs finetune, both in block mode (B=16, 1024 samples)

| steps/block | source (68001) | ent | finetuned (40000) | ent |
|---:|---:|---:|---:|---:|
| 1 | 49.64 | 3.52 | 53.85 | 4.49 |
| 2 | 60.11 | 3.97 | 52.43 | 4.64 |
| 4 | 154.55 | 4.80 | **49.28** | 5.07 |

The NFE curve flips direction: the un-finetuned source degrades with more
flow-map steps, the finetune improves. That is the direct H1 evidence.

## Raw files lost to the reboot (numbers transcribed from the run logs)

The scratchpad wipe destroyed these JSONs and two scripts
(`gen_ppl_ts.py`, `gen_ppl_gptj.py`). The measurements themselves:

**TinyStories source checkpoint (68001), full-sequence CFM, 2048 samples:**

| NFE | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| gen-PPL | 87.42 | 82.86 | **74.85** | 79.02 | 90.25 | 91.44 |
| entropy | 4.451 | 4.563 | 4.850 | 5.238 | 5.774 | 5.929 |

(An earlier 512-sample run agreed to within 0.3%: 87.16 / 82.22 / 74.31 / 78.14.)

**text8 M1 ECLD, full-sequence, 512 samples, by training step:**

| NFE | 40k | 60k | 80k | 100k |
|---:|---:|---:|---:|---:|
| 1 | 164.7 | 557.4 | 863.3 | 939.5 |
| 2 | 392.9 | 837.5 | 1164.5 | 1185.4 |
| 4 | 579.8 | 996.6 | 1269.8 | 1300.8 |
| 8 | 665.0 | 1046.6 | 1323.6 | 1352.9 |

PPL rises as training proceeds because early checkpoints were under-diverse;
entropy at NFE=1 went 2.305 -> 2.590 -> 2.659 -> 2.650 against a gold of 2.856.
