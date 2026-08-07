"""Minimal pure-PyTorch stand-in for `flash_attn`.

`semicat/net/duo.py` imports flash_attn at module level but only ever calls
`flash_attn.layers.rotary.apply_rotary_emb_torch` (itself a pure-torch reference
implementation inside flash-attn). Building the real CUDA package for sm_120 /
torch 2.7.1 is not needed just to run inference, so this shim provides that one
function verbatim (einops rewritten with plain torch ops).
"""

from flash_attn import layers  # noqa: F401

__version__ = "0.0.0+shim"
