# Training-protocol parity with the CFM reference

This repository's baselines exist to be compared against the Semicat/CFM method
in the neighbouring `semicat/` tree. That comparison is only meaningful if the
two sides are trained under the same protocol, so this document records the full
diff: what was aligned, what is deliberately still different, and what remains an
open decision.

**The reference is frozen.** Every change described here is on the baselines
side. Reference paths below are relative to `semicat/`.

Reference configs compared: TinyStories — `configs/experiment/tinystories_dit.yaml`
(the config `slurm_tinystories_a100.sbatch` launches) with
`configs/model/tinystories_dit.yaml` and `configs/data/tinystories.yaml`;
Text8 — `configs/experiment/text8_dit_lg.yaml` with `configs/model/text8_dit_lg.yaml`
and `configs/data/text8.yaml`. The reference's `lm1b` and `owt` configs have no
counterpart here and are out of scope.

## Aligned

| Aspect | Reference | Baselines now | Was |
|---|---|---|---|
| TinyStories backbone | 384 wide, 6 layers, 6 heads, MLP 1536 | same (51.1M params) | 768/12/12/3072 (169.4M) |
| Text8 backbone | 768/12/12, MLP 3072 | same (92.2M params) | already matched |
| TinyStories global batch | 32 x 4 = 128 seq = 32,768 tok/step | 128 (AR/MDLM 32x4, BD3-LM 16x8) | 256 |
| TinyStories budget | 100k steps = 3.28e9 tokens | same | 6.55e9 (2x) |
| Text8 global batch | 256 x 8 = 2048 seq | 2048 (AR/MDLM 256x8, BD3-LM 128x16) | 256 |
| Text8 budget | 500k steps = 2.6e11 tokens | same | 6.55e9 (40x fewer) |
| AdamW betas | `(0.9, 0.999)` — torch defaults, never overridden | `[0.9, 0.999]` | `[0.9, 0.95]` |
| LR schedule | `LinearLR(start_factor=1e-3, total_iters=500)`, then constant 1e-4 | linear warmup from `1e-3 x lr` over 500 steps, then constant | cosine decay to 0 |
| AR EMA | none, anywhere | `ema_decay: 0.0` | 0.9999 |
| Output-head init | zeroed (`net/duo.py` `DDiTFinalLayer`) | zeroed | Xavier-uniform |
| Runtime flags | `set_float32_matmul_precision("high")`, `cudnn.benchmark=True` | same | unset |
| TinyStories sequences | packed `[BOS, 254 tokens, EOS]`, non-overlapping | same | stride-1 windows over a flat stream, no BOS/EOS |
| TinyStories detokenization | `skip_special_tokens=True` | drops id 50256 | emitted a literal `<\|endoftext\|>` |
| Validation cadence | every 1,000 opt steps (TinyStories) / 1,250 (Text8) | same | every 10,000 |
| TinyStories validation coverage | 1,600 packed sequences | same 1,600 packed sequences | stride-1 flat-stream windows |
| Text8 validation coverage | all 19,745 stride-1 windows | same, batch 64 x up to 309 | only non-overlapping windows |
| Gen-PPL judge | GPT-2 Large, `TextMetrics.compute_mean_gen_ppl` | ported byte-for-byte, verified bit-identical | absent |

Optimizer (AdamW), lr (1e-4), weight decay (0.1), gradient clipping (1.0, norm),
precision (bf16), seed (12345), rotary embeddings (base 10000, GPT-NeoX
half-split), dropout (0.1), time-conditioning width (128), and the Text8
character vocabulary and `[351563, 20000, 19063]` split already matched.

## Deliberately different

These are intrinsic to the methods or are judged improvements; they are listed so
no one mistakes them for oversights.

- **Input representation.** The reference is a continuous-state flow model: it
  embeds a continuous `x_t` through `RMSEmbeddingLayer` (a `Linear(V→d)` plus a
  FiLM-conditioned residual MLP). The baselines embed discrete token IDs through
  `nn.Embedding`. This is what the methods *are*, and it accounts for the
  residual parameter gap (51.1M vs 52.0M on TinyStories, 92.2M vs 95.1M on
  Text8 — under 3%).
- **Time conditioning.** The reference always conditions on `(s, t)` through
  adaLN. MDLM/BD3-LM use `time_conditioning: false`, which is the SUBS
  parameterization from their own papers; forcing time conditioning on would
  make them not-MDLM.
- **Diffusion EMA.** CFM has no EMA. Published MDLM/BD3 training uses EMA
  `0.9999`, so the research diffusion configs use it for schedule search,
  validation, initialization, and inference. AR remains no-EMA. This is the
  only optimizer-state advantage deliberately taken from the BD3 protocol;
  scheduler, batch, and token budget stay CFM-matched.
- **Initialization and normalization.** Dimensions match CFM, but the shared
  baseline backbone currently applies Xavier initialization to every Linear and
  Normal(0,0.02) to embeddings. CFM leaves its main qkv/MLP/timestep Linear
  layers at PyTorch's Kaiming defaults. The official BD3 DiT also uses a
  bias-free fp32 LayerNorm and different defaults. Changing this would break
  existing MDLM checkpoint compatibility, so it is recorded rather than hidden.
- **Compilation.** Text8 CFM enables `torch.compile`; the baselines do not.
  TinyStories CFM has compile disabled. This is primarily throughput and can
  slightly change floating-point numerics, not the objective or step budget.
- **Attention masks.** AR is causal and BD3-LM uses block masks; the reference is
  fully bidirectional. Again method-intrinsic.
- **Text8 warmup units.** The reference's Text8 experiment omits
  `scheduler_interval`, so `SemicatModule`'s `"epoch"` default turns
  `LinearLR`'s 500 iterations into 500 *epochs* — roughly 85k optimizer steps of
  warmup. Its TinyStories experiment sets `scheduler_interval: step`
  explicitly. The baselines warm up over 500 optimizer steps on both datasets;
  the reference's Text8 behaviour looks unintended and is not replicated.
- **TinyStories val/test split.** The reference uses the HuggingFace validation
  split as both validation and test. The baselines hold out the second half of
  the official validation file as a separate test set. Strictly better, and it
  does not touch training.
- **Batch sampling.** The reference uses a shuffled `DataLoader` (sampling
  without replacement, epoch-based); the baselines draw random blocks with
  replacement. At these budgets the difference is immaterial.
- **Packing remainder.** The reference's `pack` map drops the tail shorter than
  254 tokens once per 1,000-story batch; `prepare_tinystories` carries it across
  batches. Worth a few hundred thousand extra tokens out of billions.
- **MAUVE.** The reference implements no MAUVE. It is added here because it was
  asked for; to compare, run it post-hoc over both repositories' saved samples
  using `bcfm_baselines.metrics.TextMetrics.compute_mauve`.

## Open decisions

1. **Text8 compute.** Matching the reference means 500k steps at global batch
   2048 — about 40x the previous Text8 setting. `configs/text8/base.yaml` now
   specifies it, but the Slurm jobs accept `MAX_STEPS` / `BATCH_SIZE` /
   `GRAD_ACCUM` overrides. If the budget is cut, cut it on *both* sides and say
   so; a shortened baseline against a full-length reference is worse than no
   comparison.
2. **Decoding strategy.** The reference decodes by argmax of the flow map — fully
   deterministic. The baselines sample with nucleus `top_p: 0.9`, which is what
   the AR/MDLM/BD3-LM papers do and what makes AR non-degenerate. This was left
   as-is because it is published practice for these models, but it is not a
   like-for-like decode: top-p mechanically lowers Gen-PPL. Either report a
   greedy arm (`strategy: greedy`, `top_p: null`) alongside, or state the
   difference wherever Gen-PPL appears.
3. **Likelihood is not comparable.** The reference's `val/loss` is a flow-matching
   cross-entropy plus a self-distillation term, not a likelihood; it has no
   validation perplexity to compare against the baselines' AR NLL or diffusion
   NELBO. Report perplexity for the baselines only, and lead the head-to-head
   with Gen-PPL and MAUVE, which both sides can produce.
4. **Sampling-step budget.** The reference sweeps `nll_steps` (1 step in the
   TinyStories config); MDLM uses 5000 denoising steps and BD3-LM 5000 per
   block. Any headline table should fix a matched NFE budget rather than
   comparing each method at its own best setting.

## Verifying the Gen-PPL port

`bcfm_baselines/metrics/text_dist.py` and `semicat/metric/text_dist.py` return
identical values, including the reference's quirks (the tail batch dropped when
the sample count is not a multiple of the scoring batch size, and scoring the
first EOS token introduced by padding). To re-check after any edit, run both on
the same list of strings with the same `context_size` and `ppl_model` and assert
exact equality — the two are pure functions of their inputs given a fixed
`transformers` version, which `requirements-a100.txt` pins to the reference's
`4.41.0`.
