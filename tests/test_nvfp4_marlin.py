"""Single-expert projections against ModelOpt dequantization + F.linear."""

import pytest
import torch
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from torch.nn import functional as F

from kernels import nvfp4_linear, prepare_nvfp4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Marlin requires CUDA"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_linear(dtype):
    torch.manual_seed(42)
    n, k = 256, 512
    original = torch.randn(n, k, device="cuda", dtype=dtype)
    quantized, scale, scale2 = NVFP4QTensor.quantize(original, block_size=16)
    raw = (quantized._quantized_data, scale, scale2)
    reference_weight = quantized.dequantize(
        dtype=dtype, scale=scale, double_scale=scale2,
        block_sizes={-1: 16}, fast=False,
    )
    prepared = prepare_nvfp4(*raw, dtype=dtype)
    host_prepared = prepare_nvfp4(*(t.cpu().pin_memory() for t in raw), dtype=dtype)
    assert all(t.is_pinned() for t in host_prepared)
    for actual, expected in zip(host_prepared, prepared):
        assert torch.equal(actual.view(torch.uint8), expected.cpu().view(torch.uint8))
    other_expert = prepare_nvfp4(
        raw[0].flip(0).cpu().pin_memory(),
        raw[1].view(torch.uint8).flip(0).view(torch.float8_e4m3fn).cpu().pin_memory(),
        (raw[2] * 2).cpu().pin_memory(),
        dtype=dtype,
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Each onload overwrites the same buffers, just as ep_plan.py reuses a slot.
        slot = tuple(torch.empty_like(t) for t in prepared)
        for step, m in enumerate((0, 1, 7, 8, 9, 33)):
            expert = other_expert if step % 2 else host_prepared
            expert_weight = reference_weight.flip(0) * 2 if step % 2 else reference_weight
            for gpu, host in zip(slot, expert):
                gpu.copy_(host, non_blocking=True)
            x = torch.randn(m, k, device="cuda", dtype=dtype) / 4
            actual = nvfp4_linear(x, *slot)
            assert actual.shape == (m, n)
            assert actual.dtype == dtype
            expected = F.linear(x.float(), expert_weight.float())
            torch.testing.assert_close(actual.float(), expected, rtol=0.025, atol=0.15)
    stream.synchronize()


def test_preparation_preserves_interleaved_rows():
    # Distinct constant rows catch transposition or accidental gate/up shuffling.
    codes = (torch.arange(128, device="cuda") % 7 + 1).to(torch.uint8)
    raw_weight = ((codes << 4) | codes)[:, None].expand(128, 64).contiguous()
    raw_scale = torch.ones(128, 8, device="cuda").to(torch.float8_e4m3fn)
    raw_scale2 = torch.tensor(0.125, device="cuda")
    prepared = prepare_nvfp4(raw_weight, raw_scale, raw_scale2, dtype=torch.float16)
    x = torch.ones(3, 128, device="cuda", dtype=torch.float16)
    output = nvfp4_linear(x, *prepared)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    expected = (lut[codes.long()] * 16).expand(3, 128)
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)
