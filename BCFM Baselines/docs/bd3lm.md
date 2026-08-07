# BD3-LM objective, mask, and sampler

This implementation follows Arriola et al., *Block Diffusion: Interpolating
Between Autoregressive and Diffusion Language Models*
([paper](https://arxiv.org/abs/2503.09573),
[official code](https://github.com/kuleshov-group/bd3lms)).

This is a controlled TinyStories/Text8 adaptation, not a byte-for-byte OWT
reproduction. The BD3 objective, mask, clipping search, resampling, EMA,
initialization from MDLM, and sampler follow the authors. By Mike's comparison
criterion, the optimizer schedule/budget instead stay CFM-matched: lr `1e-4`,
weight decay `0.1`, 500 optimizer-step warmup then constant, global batch 128
and 100k steps on TinyStories. The paper setup used lr `3e-4`, weight decay 0,
global batch 512, a longer warmup/training run, OWT/LM1B, context 1024, and a
768/12/12 backbone; its numeric curves are therefore reference shapes, not
local acceptance thresholds.

For blocks of size `B`, the sequence distribution is

```text
p(x) = product_b p_theta(x^(b) | x^(<b)),
```

and each block conditional uses the MDLM process documented in `docs/mdlm.md`.
Supported sizes are 4, 8, 16, and 32; size 16 is primary.

## Efficient training and exact attention mask

All block losses are obtained in one forward pass over `[x_t, x_0]`. Every block
has an independent low-discrepancy time and corruption draw. The official
default has no explicit time conditioning, so all AdaLN time inputs are zero;
clean conditioning keys are therefore time-independent, which is required for
exact finalized-prefix KV caching.

For noisy query block `b`, allowed keys are:

- noisy keys in block `b` (bidirectional within the active block);
- clean-copy keys in blocks strictly smaller than `b`.

It cannot see noisy earlier/future blocks, its own clean target block, or future
clean blocks. A clean-copy query sees its clean block and earlier clean blocks.
Only logits from the noisy half contribute to the MDLM NELBO.

Toy mask for `sequence_length=8`, `block_size=2` is tested exactly. `N0..N3`
denote noisy blocks and `C0..C3` clean-copy blocks:

```text
query N0 -> N0
query N1 -> N1, C0
query N2 -> N2, C0, C1
query N3 -> N3, C0, C1, C2
query C0 -> C0
query C1 -> C0, C1
query C2 -> C0, C1, C2
query C3 -> C0, C1, C2, C3
```

The boolean 16x16 pair-level mask is asserted in `tests/test_attention_masks.py`,
including explicit forbidden pairs.

## Data-driven clipped noise schedule

Research configs implement the paper and official schedule-search recipe.
Initially, block mask rates are stratified uniformly on `[epsilon, 1]`. Starting
at 5,000 optimizer steps, every validation also evaluates the full interval and every width in
`[0.5, 0.6, 0.7, 0.8, 0.9]`, shifted on a `0.05` grid. For each candidate it
sums the within-validation-batch variances of per-block NELBO estimates and
selects the minimum-variance interval. Each candidate receives an independent
diffusion-time/corruption draw, as in the official validation loop. Every
candidate variance and the selected bounds are logged, not only the winner.

Subsequent training samples mask rates from the selected interval. When the
realized masked fraction of a block falls outside that interval, its Bernoulli
corruption is resampled, matching the official recipe. An endpoint at `epsilon`
disables lower-bound resampling, and an endpoint at `1` disables upper-bound
resampling. The clipped estimator is a deliberately biased, lower-variance
training objective. Reported validation and test likelihood always use the full
configured `[epsilon, 1]` interval and are labeled as an epsilon-truncated
continuous-time NELBO estimate, never as the clipped training loss or a strict
likelihood upper bound.

Research MDLM and BD3-LM configs use the published EMA decay `0.9999`.
Schedule search, full-interval validation, sampling, and inference use EMA;
online weights are checkpointed for optimizer resume. This is an intentional
paper-faithful difference from CFM (which has no EMA), while the LR scheduler,
batch, and token budget remain CFM-matched. The selected schedule bounds are
stored in both states and restored on resume.

## MDLM initialization

The published BD3-LM experiments fine-tune a pretrained MDLM. Accordingly,
every research `bd3lm_b*.yaml` config refuses to start from scratch. First train
the repository's Text8 MDLM, then provide its checkpoint through
`training.from_pretrained` or `BD3_PRETRAIN_CHECKPOINT`. The loader requires a
matching MDLM config and strictly loads its EMA backbone; ordinary checkpoint
resume takes precedence and is also rejected if its EMA state is missing.

The single `slurm/run_all.sbatch` job wires this automatically: it trains each
dataset's MDLM first and initializes that dataset's BD3-LM from the resulting
checkpoint. Manually, the initialization checkpoint is passed through
`BD3_PRETRAIN_CHECKPOINT`:

```bash
BD3_PRETRAIN_CHECKPOINT=runs/mdlm_text8_seed12345/checkpoints/last.pt \
  python -m bcfm_baselines.train \
  --config configs/text8/bd3lm_b16.yaml --run-dir runs/bd3lm_b16_text8_seed12345
```

The official OpenWebText checkpoint cannot be loaded directly because this
Text8 adaptation has a different tokenizer, vocabulary, context, and backbone.
Smoke configs intentionally keep scratch initialization so unit tests remain
self-contained; those runs are not research results.

The official released generation checkpoints cover block sizes 4, 8, and 16.
The local size-32 config is a clearly named extension using the same algorithm,
not a claim that the paper released a size-32 checkpoint.

## Generation

Blocks are generated left-to-right. Research configs use the official
first-hitting sampler with `top_p=0.9`. If a row has `n` masks at current time
`t`, it draws `u ~ Uniform(0,1)`, advances to `t <- t * u^(1/n)`, samples clean
proposals, and uniformly reveals exactly one masked position. Therefore a block
of size `B` finishes in exactly `B` network evaluations per sequence (the first
TinyStories block takes `B-1`, because its tokenizer BOS is fixed). The
configured cap of 5000 matches the official scripts but exits as soon as the
block is complete.

Nucleus truncation also matches the official code exactly: it keeps tokens
whose cumulative probability is at most `p` (and always keeps the most likely
token), excluding the boundary-crossing token used by some standard top-p
implementations.

Categorical proposals use the paper's corrected float64 exponential/Gumbel
race rather than float32 `multinomial`; nucleus probabilities are truncated and
renormalized in float64 before the draw. TinyStories additionally follows the
official BOS behavior: the first packed token (GPT-2 id 50256) is never noised,
is excluded from evaluation NELBO, and seeds the first generated block. Text8
has no inserted BOS, so no character is excluded there.

Online Gen-PPL/MAUVE validation explicitly switches to the ancestral sampler at
64 total BD3 NFEs; a B16 first-hitting sampler cannot run in four steps because
it reveals one token per step. Offline evaluation clears those reduced-budget
overrides and restores the research first-hitting configuration.

The fixed-grid ancestral transition remains available as an explicit ablation
with `--no-first-hitting --steps-per-block N`. Its transition is
`p(mask)=s/t`; at `s=0` it reveals all remaining positions, so even one
ancestral step is a valid final transition. It is not the research-config
default.

After a block is complete it becomes immutable and is appended to the prefix.
Its time-independent K/V states are cached; only the active block is recomputed.
Cached and uncached samples are tested for exact equality under fixed seeds for
both sampler modes.

Diagnostics contain the sampler name, total NFE, per-block mask counts,
first-hitting times, entropies, wall time, step cap, cache mode, seed, and the
ancestral one-step validity flag.

## Loss curves and what to watch

The paper's conventional training curve is Figure 2 for **LM1B, block size 1**,
not TinyStories B16. It shows the uniform discrete NELBO as visibly noisy, while
the tuned/full-mask objective is smoother and closes the gap to AR. After 328M
tokens the paper reports estimator variance `1.52` for uniform NELBO versus
`0.11` for full masking; Table 1 later reports AR PPL `22.88` and tuned B1 BD3
PPL `22.88`. These numbers must not be used as TinyStories thresholds.

The authors' [project page](https://m-arriola.com/bd3lms/) gives the most useful
B16 clipping sanity check on LM1B: uniform `[0,1]` has PPL/variance
`31.72/7.62`, while `[0.3,0.8]` reaches `31.12/3.58`. Locally, after step 5,000,
the selected interval should generally reduce the logged candidate variance;
the exact interval and values can differ by dataset and backbone. The paper has
no ordinary OWT-B16 train/validation loss curve and reports no MAUVE.

Monitor `training.jsonl` or TensorBoard for NELBO/PPL and rolling variance, and
`validation.jsonl` for full-interval NELBO/PPL, every candidate variance,
selected bounds, Gen-PPL, MAUVE, entropy, sampler, and NFE. Inspect the texts in
`validation_samples/` as well: the paper explicitly warns that deceptively low
Gen-PPL can accompany low-entropy or repetitive samples. Source details are in
the [ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)
and the [developers' repository](https://github.com/kuleshov-group/bd3lms).
