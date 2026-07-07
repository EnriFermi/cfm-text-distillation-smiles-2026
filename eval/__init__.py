"""Standalone evaluation harness for BCFM.

Decoupled from the Lightning validation loop: given a checkpoint + a sampler config,
it generates samples, scores them, and writes a schema-stable ``metrics.json`` under
``results/`` (plus a ``samples.txt``). This is what every teammate reuses to produce
comparable numbers for the paper. See CLAUDE.md for the results schema.
"""
