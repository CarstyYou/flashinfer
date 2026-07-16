from __future__ import annotations

import torch

from fixture import dequantize_linear_nvfp4, unpack_e2m1


def test_unpack_e2m1_nibble_order_and_sign():
    packed = torch.tensor([[0x21, 0xBA, 0xF8]], dtype=torch.uint8)
    actual = unpack_e2m1(packed)
    expected = torch.tensor([[0.5, 1.0, -1.0, -1.5, -0.0, -6.0]])
    torch.testing.assert_close(actual, expected)


def test_linear_scale_covers_sixteen_values():
    packed = torch.full((1, 8), 0x22, dtype=torch.uint8)
    scale = torch.tensor([[2.0]], dtype=torch.float32)
    actual = dequantize_linear_nvfp4(
        packed, scale, global_scale=1.0, dtype=torch.float32
    )
    torch.testing.assert_close(actual, torch.full((1, 16), 2.0))


def test_linear_scale_reinterprets_uint8_as_e4m3_bits():
    packed = torch.full((1, 8), 0x22, dtype=torch.uint8)
    # 0x38 is the E4M3 bit pattern for 1.0, not the integer value 56.
    scale_bits = torch.tensor([[0x38]], dtype=torch.uint8)
    actual = dequantize_linear_nvfp4(
        packed, scale_bits, global_scale=1.0, dtype=torch.float32
    )
    torch.testing.assert_close(actual, torch.ones((1, 16)))
