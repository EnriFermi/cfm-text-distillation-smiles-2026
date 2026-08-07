# TinyStories revision — 2026-08-07

This archive replaces the previous benchmark section with the measured
TinyStories study. The original uploaded archive was left unchanged.

## Main changes

- Rewrote the abstract, experimental design, results, limitations, and
  conclusion around GPT-2-tokenized TinyStories at sequence length 256.
- Replaced the old plots with a joint MAUVE/latency and gen-PPL/latency figure
  covering M1-source (step 68,001), M1-200k, M2, M3/BCFM, MDLM, and BD3-LM
  over their measured NFE grids.
- Added a selected-operating-points table and reported total forward counts.
- Remeasured the uploaded step-100k EMA MDLM/BD3-LM checkpoints on the same
  H100 NVL and exact timing protocol as CFM, replacing every old A100 latency.
- Revised the cross-family conclusion: MDLM NFE 16 strictly dominates the
  best-MAUVE M3 point in latency, MAUVE, and gen-PPL under matched hardware.
- Reported CFM-family argmax decoding and canonical stochastic baseline
  sampling explicitly. No CFM sampling curve is mixed into the figure.
- Reworked long equations with aligned layouts, corrected the simplex notation
  to `Delta^{K-1}`, separated the ECLD teacher from its loss, and simplified the
  flow-map and blockwise sampling displays.
- Documented the implemented M3 Lagrangian/CSD objective separately from the
  proposed ECLD objective, and documented the non-bitwise-equivalent fast M3
  inference backend.
- Reran M2 with the same cached/compiled current-block backend as M3. Their
  matched-backend latency is within 1% at every NFE; the old roughly 10x M2
  latency gap was an inference-backend artifact.
- Corrected inconsistent bibliography entry types and removed unused template
  bibliography overrides that generated a BibLaTeX warning.

## Included evidence and reproduction

- `tinystories_quality_latency_by_nfe.csv`: all 35 plotted operating points,
  including protocol/source columns and latency percentiles.
- `scripts/plot_tinystories_paper_figure.py`: regenerates both the PDF and PNG
  figure from that CSV (requires pandas and matplotlib).
- `images/tinystories_quality_latency.pdf`: vector figure used by `main.tex`.
- `images/tinystories_quality_latency.png`: raster preview/fallback.
- `supplementary/H100_RESULTS.md`: concise matched-H100 protocol and result
  summary.
- `supplementary/baseline_h100_latency.csv`: all MDLM/BD3-LM H100 latency
  points, including percentiles and checkpoint hashes.
- `supplementary/a100_vs_h100_baseline_latency.csv`: pointwise hardware
  comparison used to audit the replacement.
- `supplementary/quality_vs_latency_all_models_argmax.png`: large dashboard
  preview containing all six model curves.

Example regeneration command:

```bash
python scripts/plot_tinystories_paper_figure.py \
  --input tinystories_quality_latency_by_nfe.csv \
  --output images/tinystories_quality_latency.pdf
```

Build the paper with `pdflatex main.tex`, `biber main`, then `pdflatex main.tex`
twice. The style keeps its original Libertine/NewTX fonts on Overleaf and has a
Latin Modern fallback for reduced local TeX installations.

## Explicit limitations retained in the paper

- Quality uses one generation seed and 512 samples per point.
- M3-fast preserves graph semantics but is not bitwise identical to the
  original FlexAttention sampler; its quality was regenerated independently.
- M2 and M3 share the same optimized sampler, but it is not bitwise identical
  to the original full FlexAttention path.
- Only block size 16 has a trained TinyStories BCFM checkpoint; there is no
  block-size-64 result.
- Latency is matched on one H100, but the methods retain their intended
  decoding modes (CFM argmax; canonical stochastic MDLM/BD3-LM samplers).

The `SMILES 2025` running footer is preserved from the supplied proceedings
template because it is venue metadata, not an experiment date.
