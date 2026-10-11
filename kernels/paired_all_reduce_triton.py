"""
Triton + symmetric-memory port of sglang's paired_all_reduce.cuh.

Four TP ranks in NVLink pairs (0, 1), (2, 3), with one PCIe hop each. Every rank
pushes its input to its NVLink peer, sums the pair, pushes the pair sum over
PCIe, and sums again.

There is no barrier: data is its own ready flag. Senders never write an all-zero
element (+0.0 is sent as -0.0), receivers spin until a slot is non-zero and then
zero it again. Only plain loads and stores cross a link, because remote
compare-and-swap (what PyTorch's signal pads use) loses updates over the PCIe hop.
"""

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

_COMMUNICATORS = {}
_MAX_BYTES = 512 * 1024
_PROGRAMS = 64
_BITS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


@triton.jit
def _mark(bits, SIGN: tl.constexpr):
    """Replace +0.0 with -0.0 so that no element in flight is all-zero bits."""
    return tl.where(bits == 0, SIGN, bits).to(bits.dtype)


@triton.jit
def _add(a, b, FLOAT: tl.constexpr):
    total = a.to(FLOAT, bitcast=True).to(tl.float32) + b.to(FLOAT, bitcast=True).to(
        tl.float32
    )
    return total.to(FLOAT).to(a.dtype, bitcast=True)


@triton.jit
def _receive(slot, mask):
    """Spin until the peer's elements have landed, then free the slot for reuse."""
    value = tl.load(slot, mask, 1, volatile=True)
    while tl.min((value != 0).to(tl.int32)) == 0:
        value = tl.load(slot, mask, 1, volatile=True)
    tl.store(slot, tl.zeros_like(value), mask)
    return value


@triton.jit
def _paired_all_reduce(
    X,
    Out,
    Own,
    Nvlink,
    Pcie,
    Counter,
    n,
    FLOAT: tl.constexpr,
    SIGN: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    X/Out are the integer-bit views of the input/output. Each workspace holds
    [phase][source link][SLOT] elements; a rank writes into its peers' workspaces
    and reads from its own. Calls alternate phases so that a peer that is one
    call ahead never overwrites elements this rank has not consumed yet.
    """
    pid = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK
    phase = tl.load(Counter + pid)
    from_nvlink = phase * 2 * SLOT
    from_pcie = from_nvlink + SLOT

    for start in range(pid * BLOCK, n, stride):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n
        local = _mark(tl.load(X + offsets, mask, 0), SIGN)
        tl.store(Nvlink + from_nvlink + offsets, local, mask)

    for start in range(pid * BLOCK, n, stride):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n
        local = tl.load(X + offsets, mask, 0)
        remote = _receive(Own + from_nvlink + offsets, mask)
        pair = _add(local, remote, FLOAT)
        tl.store(Pcie + from_pcie + offsets, _mark(pair, SIGN), mask)
        tl.store(Out + offsets, pair, mask)

    for start in range(pid * BLOCK, n, stride):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n
        pair = tl.load(Out + offsets, mask, 0)
        remote = _receive(Own + from_pcie + offsets, mask)
        tl.store(Out + offsets, _add(pair, remote, FLOAT), mask)

    tl.store(Counter + pid, phase ^ 1)


class PairedAllReduce:
    def __init__(self, group):
        rank = dist.get_rank(group)
        workspace = symm_mem.empty(4 * _MAX_BYTES, dtype=torch.uint8, device="cuda")
        workspace.zero_()
        torch.cuda.synchronize()
        self.handle = symm_mem.rendezvous(workspace, group)
        # (own, NVLink peer, PCIe peer) workspaces, viewed per element width.
        self.workspaces = {
            bits: [
                self.handle.get_buffer(peer, (4 * _MAX_BYTES // bits.itemsize,), bits)
                for peer in (rank, rank ^ 1, 3 - rank)
            ]
            for bits in _BITS.values()
        }
        self.counter = torch.zeros(_PROGRAMS, dtype=torch.int32, device="cuda")
        dist.barrier(group=group)

    def all_reduce(self, x):
        bits = _BITS[x.dtype]
        output = torch.empty_like(x)
        size = x.numel() * x.element_size()
        # The spin in _receive waits on a whole block, so small blocks poll best.
        warps = 1 if size <= _PROGRAMS * 32 * 16 * 4 else 4
        _paired_all_reduce[(_PROGRAMS,)](
            x.view(bits),
            output.view(bits),
            *self.workspaces[bits],
            self.counter,
            x.numel(),
            FLOAT=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32,
            SIGN=-(1 << (8 * bits.itemsize - 1)),
            SLOT=_MAX_BYTES // bits.itemsize,
            BLOCK=warps * 32 * 16 // bits.itemsize,
            num_warps=warps,
        )
        return output


def all_reduce(x, group):
    size = x.numel() * x.element_size()
    if (
        x.is_cuda
        and x.dtype in _BITS
        and x.is_contiguous()
        and size <= _MAX_BYTES
        and dist.get_world_size(group) == 4
        and os.getenv("INKLING_PAIRED_ALLREDUCE", "1") == "1"
    ):
        if group not in _COMMUNICATORS:
            _COMMUNICATORS[group] = PairedAllReduce(group)
        return _COMMUNICATORS[group].all_reduce(x)
    dist.all_reduce(x, group=group)
    return x
