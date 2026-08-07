# Current M2 vs upstream cached M2

Дата: 2026-07-31

## Короткий вывод

На проверенной точке `NFE=4, argmax` текущая masked-2L реализация и M2 из
`origin/feature/m2-infer-experiments@2e21c57` **семантически согласованы, но не
побитово идентичны**.

- Структурной ошибки в текущем attention/sampling graph не обнаружено: один и
  тот же checkpoint strict-loadится в обе сети, same-state logits близки, а
  малые end-to-end probes совпадают полностью.
- На полном fair run совпало `127850/131072 = 97.5418%` token IDs и полностью
  совпали `385/512` последовательностей. Следовательно, утверждать literal
  identity нельзя.
- Агрегатное качество почти одинаковое. Upstream получил GPT-J PPL `149.893`
  против `150.635` у current; парная разница мала (`-0.493%` в пользу upstream).
- Главное практическое отличие — compute: upstream с clean-prefix KV cache занял
  `3.047 с`, current с полным masked `2L` — `34.2 с`. Наблюдаемый выигрыш около
  `11.2x` в этом end-to-end run (примерно `10–11x` с учётом прежнего
  воспроизводимого current timing).

Итого: **текущий M2 выглядит корректным как математический oracle той же
семантики, но upstream runtime существенно эффективнее и должен быть
референсом для production inference.** Если нужна побитовая идентичность с
upstream, текущая реализация её не даёт.

## Fair protocol

| Параметр | Значение |
|---|---|
| Dataset / tokenizer | TinyStories / GPT-2 |
| Checkpoint | `best.ckpt`, step `68001` |
| Checkpoint SHA-256 | `a7a85b5626c9bd2c743a8773827b1ea1f24de831186490b934ceecab4a41ba6f` |
| Checkpoint type | M1, trained `block_size=None`, Gaussian prior |
| Sequence / block | `L=256`, `B=16` |
| Integration | uniform Euler, `NFE/block=4` |
| Discretization | argmax |
| RNG / batching | seed `0`, `512` samples, batch `64` |
| Arithmetic / device | fp32, NVIDIA H100 NVL, Torch `2.7.1+cu126` |
| Current revision | `07ffd555dbbbd544b22933b5a23e9ab64d633dea` |
| Upstream revision | `2e21c57704ab94e131e3350ffd8286632f61a7b8` |

Upstream запускался из чистого detached worktree. Его sampler/model были
импортированы непосредственно оттуда; четыре relevant source-файла проверены
SHA-256 против Git commit. Текущие relevant production-файлы также побайтно
совпадают с `07ffd55`.

## Чем реализации отличаются

Current:

1. Загружает M1 weights strict в state-compatible `BlockDIT`.
2. На каждом Euler jump строит `[clean; noisy]` длины `2L=512`.
3. Пересчитывает полный masked graph через FlexAttention.
4. Берёт logits только текущего noisy block.

Upstream:

1. Оставляет исходный scalar-time `DIT` с теми же M1 weights.
2. Считает только текущий noisy block через `forward_block`.
3. Предыдущие финализированные clean blocks хранит в layer-wise KV cache.
4. После каждого не последнего блока один раз вызывает `encode_clean(s=t=1)`.

Dependency graph один: noisy block видит собственный noisy block и только
предыдущий clean prefix. Различаются tensor shapes, attention kernels и порядок
floating-point reductions.

## Token-level результат

| Проверка | Результат |
|---|---:|
| Matching token IDs | `127850 / 131072` (`97.5418%`) |
| Different token IDs | `3222` (`2.4582%`) |
| Полностью совпавшие sequences | `385 / 512` (`75.20%`) |
| Mean Hamming distance | `6.293 / 256` tokens |
| Median Hamming distance | `0` |
| Maximum Hamming distance | `142` |

Agreement по блокам уменьшается с `99.8535%` в block 0 до `95.8374%` в block
15. Это ожидаемый профиль накопления: редкий ранний argmax flip меняет clean
prefix и может разветвить дальнейшую траекторию.

Fresh current-run побитово воспроизвёл прежний `dumps/ts_M2_source` для этой
точки. Оба JSON точно декодируются из соответствующих NPZ (`0` decode
mismatches), обе стороны дали `512/512` уникальных строк, NaN/invalid token IDs
не обнаружено.

## Same-state discriminator

Для отделения semantic mismatch от numerics один M1 state был strict-loaded в
обе сети, после чего logits сравнивались на одинаковых clean/noisy states для
blocks `0, 1, 8, 15`.

| Float32 matmul mode | Max abs logit delta | Argmax disagreement | Mini e2e |
|---|---:|---:|---:|
| `highest` | `1.812e-5` | `0/64` | `0/2048` tokens |
| `high` (production-like TF32) | `0.004091` | `0/64` | `0/2048` tokens |

Mini e2e покрывает `argmax/sample × NFE 1/2`, batch `2`. Intervention
`highest → high` увеличил максимальное расхождение logits более чем в 200 раз,
не меняя граф или weights. Вместе с full-run blockwise accumulation это
поддерживает механизм **kernel/layout/TF32 rounding → редкие near-tie token flips
→ propagation через clean prefix**. Для каждого из 3222 full-run flips margin
отдельно не сохранён, поэтому эта последняя детализация остаётся наиболее
поддержанной причиной, а не формальным доказательством для каждого токена.

## Common quality scoring

Обе выборки прошли один и тот же current scorer, один reference dump,
GPT-2-large MAUVE featurizer и GPT-J-6B judge.

| Метрика | Current masked-2L | Upstream KV | Upstream − current |
|---|---:|---:|---:|
| GPT-J gen-PPL ↓ | `150.63515` | `149.89269` | `-0.74245` |
| MAUVE ↑ | `0.005805` | `0.005705` | `-0.000100` |
| Token entropy | `3.990724` | `3.988345` | `-0.002379` |
| Seq-rep-2 ↓ | `0.128271` | `0.128914` | `+0.000643` |
| JS-1gram ↓ | `0.171440` | `0.171872` | `+0.000432` |
| JS-2gram ↓ | `0.410415` | `0.410715` | `+0.000300` |

Парный bootstrap по per-sequence GPT-J NLL (`20,000` resamples) дал relative
PPL delta upstream-current `-0.493%`, 95% CI `[-0.990%, -0.028%]`, вероятность
upstream lower `0.9814`. Это небольшой сдвиг на данной парной выборке, а не
основание утверждать общее преимущество на других NFE/seeds.

Важно: **обе генерации на NFE=4 качественно плохи** — тексты часто
несогласованные, повторяющиеся и содержат mojibake, а MAUVE около `0.006` крайне
низок. Эксперимент валидирует соответствие двух implementations; он не
валидирует checkpoint/NFE=4 как хороший baseline.

## Runtime

| Реализация | Flow forwards / sequence | Extra cache builds | 512 samples |
|---|---:|---:|---:|
| Current full masked `2L` | `64` | `0` | `34.2 с` |
| Upstream current-block KV | `64` | `15` | `3.047 с` |

Raw forward counts нельзя сравнивать как FLOPs: upstream имеет больше вызовов,
но они работают на block/prefix cache, тогда как current каждый раз пересчитывает
полные 512 positions. Timing синхронизирован и исключает model load, но это по
одному основному timed run на arm, не p50/p90 microbenchmark.

## Validity checks и исключённые confounders

- Один checkpoint и полный SHA-256: совпадает.
- Один seed, batch size, sample count, schedule, prior, dtype и device: совпадают.
- Один state namespace; strict-load missing/unexpected keys: `0/0` для обеих сетей.
- Stale current artifact: исключён fresh bitwise reproduction.
- Смешанный импорт current/upstream: исключён проверкой import paths и Git hashes.
- JSON/NPZ mismatch, duplicate collapse, invalid IDs: не обнаружены.
- Revision-local tests: current `26/26`, upstream `21/21` passed.
- Scorer mismatch: исключён одним scorer process/config; current metrics также
  воспроизвели ранее сохранённую строку.

Revision-local tests сами по себе не являются cross-revision equivalence test;
для этого служат actual-checkpoint same-state probes выше.

## Граница вывода

Основной quality/token run устанавливает результат только для `argmax, NFE=4,
seed=0, batch=64, fp32, H100`. Нельзя автоматически переносить процент token
agreement на другие NFE, categorical sampling, bf16, другие GPUs или checkpoints.
Mini probes на `sample` и NFE `1/2` подтверждают согласованность, но слишком малы
для quality claims.

## Рекомендация

Для production M2 стоит перенести минимальный upstream cache API
(`encode_clean`, `forward_block`, cached loop) под отдельным sampler name и
оставить нынешний full masked path как oracle/debug reference. Всю ветку целиком
merge/cherry-pick делать не стоит: она далеко разошлась с current и содержит
много несвязанных изменений.

Перед заменой закрепить regression test на actual checkpoint:

- shared stored priors / categorical uniforms;
- per-block logits/probability tolerances отдельно для `highest` и `high`;
- token comparison на нескольких NFE и обоих discretizers;
- synchronized warmup + median/p10/p90 latency и peak memory.

Production-код в рамках этого сравнения не менялся; добавлены только evidence
artifacts.

## Артефакты

- [comparison_summary.json](comparison_summary.json) — полный machine-readable итог.
- [current_m2_nfe4.json](current_m2_nfe4.json) и
  [current_m2_nfe4.tokens.npz](current_m2_nfe4.tokens.npz).
- [upstream_m2_nfe4.json](upstream_m2_nfe4.json) и
  [upstream_m2_nfe4.tokens.npz](upstream_m2_nfe4.tokens.npz).
- [scores.json](scores.json) — common scorer output.
- [paired_judge_bootstrap.json](paired_judge_bootstrap.json) — paired GPT-J NLL и CI.
- [probe_results.json](probe_results.json) — `highest` same-state probe.
- [probe_results_high.json](probe_results_high.json) — production-like `high` probe.
- [run_upstream_exact.py](run_upstream_exact.py) — orchestration вокруг неизменённого upstream code.
- [run_same_state_probe.py](run_same_state_probe.py) — воспроизводимый cross-revision probe.
- [current_tests.xml](current_tests.xml) и [upstream_tests.xml](upstream_tests.xml).
