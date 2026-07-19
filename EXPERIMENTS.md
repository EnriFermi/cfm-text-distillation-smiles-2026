# Experiment registry

Append a row when you launch a run. Human-readable status here; machine-readable numbers
live in `results/<exp>/metrics.json` (aggregated by `python -m eval.aggregate`). The full
design — what each experiment tests and its decision rule — is in
[docs/experiment_plan.md](docs/experiment_plan.md). Status: `planned`/`running`/`done`/`dropped`.

## Claim → experiment (see docs/experiment_plan.md for details)
`C1/H2`→E1,E13 · `C2/H1`→E2,E2b · `C3` sweet-spot→E3 · `C4` B×steps→E4 · `C5` collapse→E5 ·
`C6` exposure→E6 · `C7` cache/length→E12,E13 · `C8` ECLD→E7.

## Run tracking

| exp (E#) | config / command | model | B | steps/blk | seeds | owner | status | comet |
|---|---|---|---|---|---|---|---|---|
| E0-ref | `sampler=gold n_samples=512` | data | – | – | 0 | misha | planned | |
| E0/E1 | `sampler=full_cfm sampler.steps=1,2,4,8,16` | M1 | 256 | 1–16 | 0,1,2 | misha | planned | |
| E1/E3/E4 | `sampler=bcfm_infer` (B×steps grid) | M2 | 1–256 | 1,2,4 | 0,1,2 | — | planned | |
| E6 | `sampler.prefix=generated,gold` | M2 | 16 | 1,2 | 0,1,2 | — | planned | |
| E10 | `sampler.schedule=... exp_name=...` | M2 | 16 | 2 | 0 | — | planned | |
| E11 | `sampler.discretize=argmax,sample` | M2 | 16 | 1 | 0,1,2 | — | planned | |
| — train — | `experiment=cfm_text8_baseline` | M1 | – | – | 12345 | misha | planned | |
| E2b | continue M1 for M3's finetune budget (M1+) | M1+ | – | – | 12345 | — | planned (P1) | |
| E7 | `experiment=bcfm_train_text8 model.sd_type=ecld,lag,none` | M3 | 16 | – | 0 | — | planned (P1) | |
| E8 | `... model.sd_prop=0,0.25,0.5` | M3 | 16 | – | 0 | — | planned (P1) | |
| E9 | `... model.prior_type=gaussian,discunif` | M1/M3 | – | – | 0 | — | planned (P1) | |
| E1/E2/E12 | eval each trained M3 (`sampler=bcfm_train`) | M3 | 4,8,16 | 1,2,4 | 0,1,2 | — | planned (P1) | |

Priority (docs/experiment_plan.md §6): **must-have** E0,E1,E2,E3,E5,E13 · should-have E4,E6,E12 ·
nice-to-have E7–E11.

## Deliverables → artifacts (produced by eval/aggregate.py)
- **Fig.1** gen-PPL vs network forwards/token (H2) — `results/figures/fig1_nfe_vs_genppl.png`
- **Fig.2** gen-PPL vs block size (sweet spot) — `fig2_blocksize.png`
- **Fig.3** gen-PPL vs steps/block — `fig3_steps.png`
- **Fig.4** entropy per block index (collapse) — `fig4_entropy_per_block.png`
- **Tab. H1** M3 vs M2 paired — from `results/summary.csv`
- **Fig.5 / E13** wall-clock tokens/sec — separate timing run (see plan §5).
