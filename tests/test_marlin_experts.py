"""Routed NVFP4 experts through the Marlin kernels, against an fp64 dequantized reference.

Run with ``uv run --frozen -m pytest tests/test_marlin_experts.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from kernels import ExpertCache, GroupedExperts
from kernels.nvfp4_marlin import marlin_utils
from utils.checkpointing import pack_experts_for_marlin

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Marlin kernels require CUDA"
)

HIDDEN, INTERMEDIATE, EXPERTS, GPU_EXPERTS, TOP_K = 4096, 2048, 8, 3, 6
NAMES = [
    projection + suffix
    for projection in ("gate_up_proj", "down_proj")
    for suffix in ("", "_scale", "_scale2")
]


@pytest.fixture(scope="module")
def experts():
    """Checkpoint-layout banks (first experts on GPU, the rest pinned) and dense fp64 weights."""
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(0)
    state = {name: {} for name in NAMES}
    dense = {"gate_up_proj": {}, "down_proj": {}}
    for expert in range(EXPERTS):
        for name, shape in (
            ("gate_up_proj", (2 * INTERMEDIATE, HIDDEN)),
            ("down_proj", (HIDDEN, INTERMEDIATE)),
        ):
            weight = torch.randn(shape, generator=generator) * 0.02
            quantized = marlin_utils.nvfp4_quantize(weight.to(device))
            dense[name][expert] = quantized.dequantize()
            for suffix, value in (
                ("", quantized.packed),
                ("_scale", quantized.block_scale),
                ("_scale2", quantized.global_scale),
            ):
                value = value.contiguous()
                state[name + suffix][expert] = (
                    value if expert < GPU_EXPERTS else value.cpu().pin_memory()
                )
    pack_experts_for_marlin(state, torch.bfloat16, device)
    return state, dense


def build(state, num_slots):
    device = torch.device("cuda", torch.cuda.current_device())
    grouped = GroupedExperts(
        set(range(EXPERTS)), HIDDEN, INTERMEDIATE, torch.bfloat16, device, True, F.silu
    )
    return grouped, ExpertCache(state, NAMES, num_slots, grouped)


def reference(dense, x, indices, weights):
    output = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for token in range(x.shape[0]):
        for expert, weight in zip(indices[token].tolist(), weights[token].tolist()):
            gate, up = (
                (x[token].double() @ dense["gate_up_proj"][expert].T).view(-1, 2).T
            )
            output[token] += weight * (
                (F.silu(gate) * up) @ dense["down_proj"][expert].T
            )
    return output


def routing(tokens, generator):
    indices = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(tokens)]
    )
    weights = torch.rand(tokens, TOP_K, generator=generator).bfloat16()
    return indices.cuda(), weights.cuda()


def run(grouped, cache, x, indices, weights):
    output = torch.zeros_like(x, dtype=torch.float32)
    cache.prepare(indices)
    grouped.forward(x, indices, weights, output)
    return output


def assert_close(output, expected):
    error = (output.double() - expected).abs().max() / expected.abs().max()
    assert error < 2e-2, error


def test_packing_keeps_placement(experts):
    state, _ = experts
    for name in NAMES:
        for expert, value in state[name].items():
            assert value.is_cuda == (expert < GPU_EXPERTS)
            assert value.is_cuda or value.is_pinned()
    assert state["gate_up_proj"][0].shape == (HIDDEN // 16, 4 * INTERMEDIATE)
    assert state["down_proj_scale"][0].shape == (INTERMEDIATE // 16, HIDDEN)
    assert state["down_proj_scale2"][0].shape == (2,)


@pytest.mark.parametrize("tokens", [1, 2, 40])
def test_forward_matches_reference(experts, tokens):
    state, dense = experts
    # Three slots for five pinned experts: prefill also runs experts with no slot.
    grouped, cache = build(state, num_slots=3 if tokens > 1 else TOP_K)
    generator = torch.Generator().manual_seed(tokens)
    for _ in range(3):
        x = torch.randn(tokens, HIDDEN, generator=generator).bfloat16().cuda()
        indices, weights = routing(tokens, generator)
        output = run(grouped, cache, x, indices, weights)
        assert_close(output, reference(dense, x, indices, weights))


def test_token_matches_rows_path(experts):
    """The grouped decode launch and the host-dispatched path agree bit for bit."""
    state, _ = experts
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, HIDDEN, generator=generator).bfloat16().cuda()
    indices, weights = routing(1, generator)
    grouped, cache = build(state, TOP_K)
    single = run(grouped, cache, x, indices, weights)
    grouped, cache = build(state, TOP_K)
    doubled = run(
        grouped, cache, x.repeat(2, 1), indices.repeat(2, 1), weights.repeat(2, 1)
    )
    assert torch.equal(single[0], doubled[0]) and torch.equal(single[0], doubled[1])


def test_decode_graph_follows_routing(experts):
    """A captured forward replays correctly for routes and inputs chosen after capture."""
    state, dense = experts
    grouped, cache = build(state, TOP_K)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, HIDDEN, generator=generator).bfloat16().cuda()
    indices, weights = routing(1, generator)
    output = torch.zeros_like(x, dtype=torch.float32)

    def step():
        output.zero_()
        cache.prepare(indices)
        grouped.forward(x, indices, weights, output)

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(8):
        x.copy_(torch.randn(1, HIDDEN, generator=generator))
        new_indices, new_weights = routing(1, generator)
        indices.copy_(new_indices)
        weights.copy_(new_weights)
        graph.replay()
        assert_close(output, reference(dense, x, indices, weights))
