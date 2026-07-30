import torch

from bcfm_baselines.net.text_transformer import (
    bd3_training_attention_mask,
    block_causal_attention_mask,
)


def test_toy_bd3_training_mask_sequence_8_block_2() -> None:
    mask = bd3_training_attention_mask(8, 2)
    assert mask.shape == (16, 16)
    # Noisy block 0 is bidirectional only within noisy keys 0:2.
    assert mask[0, 0] and mask[0, 1]
    assert not mask[0, 2]
    assert not mask[0, 8]  # never its own clean copy
    # Noisy block 2 sees its own noisy block and clean blocks 0 and 1.
    assert mask[4, 4] and mask[4, 5]
    assert mask[4, 8] and mask[4, 11]
    assert not mask[4, 12]  # own clean block forbidden
    assert not mask[4, 6]   # future noisy block forbidden
    # Clean block 2 sees clean blocks 0..2, never noisy tokens or future clean.
    assert mask[12, 8] and mask[12, 13]
    assert not mask[12, 0]
    assert not mask[12, 14]
    # Every query has at least one legal key.
    assert mask.any(dim=-1).all()


def test_generation_block_mask() -> None:
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(block_causal_attention_mask(8, 2), expected)

