# MDLM objective and sampler

This implementation follows the absorbing-state SUBS formulation of
Sahoo et al., *Simple and Effective Masked Diffusion Language Models*
([paper](https://arxiv.org/abs/2406.07524),
[official code](https://github.com/kuleshov-group/mdlm)).

## Forward corruption

The data vocabulary has 27 Text8 characters and the model-only absorbing state
is `m = 27`. For continuous time `t ~ Uniform(epsilon, 1)` the survival schedule
is

```text
alpha(t) = 1 - t.
q(x_t | x_0) = alpha(t) delta[x_0] + (1-alpha(t)) delta[m].
```

Corruption is independent across positions. Time values use the official
low-discrepancy/antithetic minibatch stratification by default. For a minibatch
of size `N`, independent `r_i ~ Uniform(0,1)` values are transformed as

```text
u_i = (r_i / N + i / N) mod 1
t_i = epsilon + (1-epsilon) u_i.
```

Thus each item occupies a distinct equal-width stratum; no clamping creates an
endpoint point mass. Padding, if introduced by a future dataset, is excluded
through `attention_mask`.

## Prediction and continuous-time NELBO

The Transformer consumes `x_t` and predicts the clean character distribution.
As in the official default configuration, `time_conditioning: false`: the
AdaLN pathway receives a fixed zero-time embedding rather than the sampled
`t`. Explicit time conditioning remains available as a documented ablation.
SUBS is enforced conceptually:
probability of a clean target being `[MASK]` is zero and already-clean tokens
are carried unchanged by the reverse process. Only masked positions contribute.

For the linear survival schedule, `-alpha'(t)/(1-alpha(t)) = 1/t`, so the
Rao-Blackwellized objective in nats per dataset token is

```text
L = E_t E_{x_t|x_0} [
      sum_i 1[x_t^i=m] (-log p_theta(x_0^i | x_t,t)) / t
    ] / number_of_valid_tokens.
```

The denominator is all valid tokens, not only masked tokens. This gives the
official continuous-time Monte Carlo loss per token. `epsilon=1e-3` avoids the
numerical singularity; the masked indicator cancels the `1/t` factor in
expectation. Because time is sampled only on `[epsilon, 1]`, the reported scalar
is explicitly labeled an epsilon-truncated NELBO estimate, not a mathematically
strict upper bound on the full `t in [0,1]` likelihood. Validation and test
aggregation weight every batch metric by its valid-token count. Evaluation
resets the Torch RNG from the requested seed before drawing times and
corruptions, making the single-sample estimate reproducible. The reported BPC
is likewise named `bits_per_character_estimate`.

## Official DDPM-cache reverse sampler

Sampling starts from all masks at `t=1`. The research config uses the official
5000-step grid down to `sampling_epsilon=1e-5`. For `t > s >= epsilon`, an
already unmasked token is copied and a masked token follows

```text
p(x_s = m | x_t=m) = s/t
p(x_s = v | x_t=m) = (1-s/t) p_theta(v | x_t,t), v != m.
```

If a transition changes no tokens and explicit time conditioning is disabled,
the clean-prediction logits are cached exactly as in the official
`ddpm_cache` predictor. After reaching epsilon, the official deterministic
noise-removal pass fills any remaining masks with the denoiser argmax while
preserving already-clean tokens. The sampler records actual NFE, configured
diffusion steps, mask count, predictive entropy, epsilon, cache/noise-removal
settings, wall time, schedule, and seed.

The research AR, MDLM, and BD3-LM sampling configs all use the official
generation setting `top_p=0.9`. Their nucleus mask excludes the first token
whose cumulative mass crosses `p` (while always retaining the top token),
matching the repository implementation rather than the common boundary-
inclusive variant.

## Differences from the official large-language implementation

- Text8 uses the reference 27-character vocabulary and a much smaller context.
- The noise schedule is the schedule-invariant linear-survival form rather than
  materialized transition matrices.
- Evaluation is offline Text8 epsilon-truncated NELBO/BPC estimates and
  distribution statistics; external
  GPT Gen-PPL is not a required metric.
- The Transformer width/depth follows the CFM comparison backbone.
- Training maintains an EMA of parameters; validation, checkpoints used for
  inference, sampling, and evaluation use EMA weights by default.
