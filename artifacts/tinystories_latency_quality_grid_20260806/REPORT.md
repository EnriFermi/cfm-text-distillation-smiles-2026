# TinyStories MAUVE/gen-PPL vs batch=1 latency

Date: 2026-08-06 UTC

## Outcome

The complete existing TinyStories quality grid was joined with batch-size-one
latency: 4 model/sampler variants × 2 discretizations × 6 NFE values = 48 points.
All rows are unique and finite, and every latency row maps to an existing quality
row by `(label, point)`.

The strongest low-latency model is M1 at step 200000.  Its argmax NFE=32 point
reaches MAUVE 0.657744 at 133.788 ms median latency.  M3 reaches the overall best
argmax MAUVE, 0.797925, at NFE=16 and 1830.439 ms, about 13.7× the latency.  In
sample mode, M1 step 200000 NFE=32 (MAUVE 0.650006 at 130.488 ms) dominates every
measured M3 point in both latency and MAUVE.

## Protocol

- GPU: NVIDIA H100 NVL.
- dtype: float32.
- sequence length: 256.
- latency batch size: 1.
- seed: 0.
- warmup: 5 runs per point.
- measured repetitions: 30 per point.
- reported latency: median, p10 and p90 wall-clock milliseconds.
- each measured interval is bracketed by CUDA synchronization.
- quality sample count: 512 per point.
- NFE: 1, 2, 4, 8, 16, 32.
- discretization: argmax and categorical sample.

The timing functions `_sync` and `_measure_sequence_latency_ms` were loaded and
executed directly from the AST of `eval/run_eval.py` at upstream commit
`2e21c57704ab94e131e3350ffd8286632f61a7b8`.  The stored source snapshot has
SHA-256 `120e2e53826dc540d1c7372049c1582c8bd9af679c4412489021b0ebdd3bce71`, identical
to `git show origin/feature/m2-infer-experiments:eval/run_eval.py`.

Model loading and sampling use the current quality-dump implementation.  This is
necessary for a valid join: the full upstream branch contains an older KV-cache
M2/M3 sampler, while the stored MAUVE/gen-PPL rows use the current identical
`block_causal` loop for M2 and M3.  Only the weights differ between those two
quality variants.

## Checkpoints and quality sources

- M1 source (nominally called the 100k baseline):
  `baseline/tinystories/NEW/tinystories_a100/checkpoints/best.ckpt`, actual stored
  `global_step=68001`.
- M2: the same source M1 checkpoint loaded into the block architecture, B=16.
- M3: `logs/train/2026-07-28_12-43-24_12345_local/checkpoints/step_0100000.ckpt`,
  `global_step=100000`, B=16.
- M1 step 200000:
  `logs/train/tinystories_cfm_resume144001_to200000_nfe32_20260806/checkpoints/final/step_00200000_full_state.ckpt`.
- Existing quality metrics:
  `results/ts_M1_M2_M3_scores.json` and
  `results/ts_M1_step200000_full_scores.json`.

## Argmax: median latency ms / MAUVE

| NFE | M1 source | M2 block | M3 block | M1 step 200k |
|---:|---:|---:|---:|---:|
| 1 | 4.780 / 0.0058 | 112.758 / 0.0049 | 112.548 / 0.0080 | 4.875 / 0.0066 |
| 2 | 11.464 / 0.0071 | 222.505 / 0.0053 | 242.929 / 0.0095 | 9.021 / 0.0081 |
| 4 | 16.671 / 0.0138 | 440.654 / 0.0058 | 451.758 / 0.0391 | 17.278 / 0.0196 |
| 8 | 32.622 / 0.0654 | 882.778 / 0.0058 | 911.942 / 0.1570 | 34.279 / 0.1591 |
| 16 | 64.662 / 0.2381 | 1793.504 / 0.0146 | 1830.439 / 0.7979 | 67.990 / 0.5229 |
| 32 | 125.735 / 0.5067 | 3538.574 / 0.0262 | 3673.105 / 0.6271 | 133.788 / 0.6577 |

## Categorical sample: median latency ms / MAUVE

| NFE | M1 source | M2 block | M3 block | M1 step 200k |
|---:|---:|---:|---:|---:|
| 1 | 4.885 / 0.0062 | 117.033 / 0.0046 | 132.985 / 0.0072 | 5.105 / 0.0078 |
| 2 | 8.786 / 0.0073 | 227.144 / 0.0050 | 261.398 / 0.0111 | 9.277 / 0.0080 |
| 4 | 16.633 / 0.0100 | 462.721 / 0.0064 | 517.837 / 0.0308 | 17.945 / 0.0219 |
| 8 | 32.444 / 0.0266 | 908.195 / 0.0075 | 1031.344 / 0.2882 | 34.426 / 0.2951 |
| 16 | 63.982 / 0.1852 | 1776.135 / 0.0141 | 1849.754 / 0.5978 | 66.999 / 0.4588 |
| 32 | 126.642 / 0.4089 | 3552.119 / 0.0167 | 3594.005 / 0.5777 | 130.488 / 0.6500 |

## Review and caveats

- The M2 curve is entirely dominated by full-sequence M1: it is slower and has
  lower MAUVE at every useful operating point.
- M3 argmax NFE=32 is internally dominated by M3 NFE=16: latency doubles while
  MAUVE falls from 0.797925 to 0.627053.
- The present block-causal implementation is expensive: at NFE=16, M3 argmax is
  about 26.9× slower than M1 step 200000 at the same NFE.  This is measured wall
  time, not inferred from NFE.
- Checkpoints with the same runtime graph show small run-order differences.  Most
  are a few percent; low-NFE sample points differ by as much as 15%.  Do not
  interpret these small cross-checkpoint differences as weight-dependent speed.
  The large full-CFM versus block-causal gap is much larger than this noise.
- M1 source argmax NFE=2 has a relatively broad p10–p90 interval
  (8.804–14.261 ms); the plot includes horizontal p10/p90 error bars.
- The MAUVE and gen-PPL values were not recomputed during this run; they are the
  already reviewed 512-sample metrics joined to the exact sampler settings.

## Artifacts

- `latency_quality_grid.json`: full metadata and 48 rows.
- `latency_quality_grid.csv`: plotting/analysis table.
- `mauve_vs_latency.png`: reviewed MAUVE/latency curves.
- `genppl_vs_latency.png`: reviewed gen-PPL/latency curves.
- `run.log`: complete benchmark log.
- `upstream_run_eval_2e21c57.py`: exact upstream timing-source snapshot.
- `../../eval/measure_quality_grid_latency.py`: reproducible harness.
