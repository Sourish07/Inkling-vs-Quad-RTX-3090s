"""
GPU cache planning and onload from CUDA-mapped pinned checkpoint tensors.
"""

import ctypes

import torch
import triton
import triton.language as tl


def mapped_pointer(tensor):
    """
    Resolve pinned host storage once; its owning tensor must remain alive.
    """
    if tensor.is_cuda:
        return tensor.data_ptr()
    assert tensor.is_pinned() and tensor.is_contiguous()
    driver = ctypes.CDLL("libcuda.so.1")
    pointer = ctypes.c_uint64()
    status = driver.cuMemHostGetDevicePointer_v2(
        ctypes.byref(pointer), ctypes.c_void_p(tensor.data_ptr()), 0
    )
    if status:
        raise RuntimeError(f"cuMemHostGetDevicePointer failed: {status}")
    return pointer.value


@triton.jit
def _plan(
    Indices,
    Sources,
    Resident,
    Destinations,
    SlotExperts,
    Ages,
    Clock,
    CopyExperts,
    Pointers,
    Experts,
    Counts,
    Rows,
    START: tl.constexpr,
    LOCAL: tl.constexpr,
    SLOTS: tl.constexpr,
    STRIDE: tl.constexpr,
    ROUTES: tl.constexpr,
    CAPACITY: tl.constexpr,
    E: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
):
    e = tl.arange(0, E)
    s = tl.arange(0, S)
    r = tl.arange(0, R)
    ids = tl.load(Indices + r, r < ROUTES, -1)
    matches = (
        (e[:, None] < LOCAL)
        & (r[None, :] < ROUTES)
        & (ids[None, :] == START + e[:, None])
    )
    counts = tl.sum(matches.to(tl.int32), 1)
    active = counts > 0
    resident = tl.load(Resident + e, e < LOCAL, 0) != 0
    cached = tl.load(SlotExperts + s, s < SLOTS, -1)
    hits = (cached[None, :] == e[:, None]) & (s[None, :] < SLOTS)
    slot = tl.sum(tl.where(hits, s[None, :] + 1, 0), 1) - 1
    missing = active & ~resident & (slot < 0)
    protected = tl.sum((hits & active[:, None]).to(tl.int32), 0) > 0
    age = tl.load(Ages + s, s < SLOTS, 0)
    # Rank unprotected slots by age, with slot ID breaking ties. Active hits
    # cannot be evicted, including when prefill uses more experts than fit.
    eligible = (s < SLOTS) & ~protected
    earlier = (age[None, :] < age[:, None]) | (
        (age[None, :] == age[:, None]) & (s[None, :] < s[:, None])
    )
    rank = tl.sum((earlier & eligible[None, :]).to(tl.int32), 1)
    missing_rank = tl.cumsum(missing.to(tl.int32)) - 1
    assign = (
        missing[:, None] & eligible[None, :] & (missing_rank[:, None] == rank[None, :])
    )
    new_slot = tl.sum(tl.where(assign, s[None, :] + 1, 0), 1) - 1
    slot = tl.where(new_slot >= 0, new_slot, slot)
    copy_expert = tl.sum(tl.where(assign, e[:, None] + 1, 0), 0) - 1
    cached = tl.where(copy_expert >= 0, copy_expert, cached)
    tick = tl.load(Clock) + 1
    used = protected | (copy_expert >= 0)
    tl.store(SlotExperts + s, cached, s < SLOTS)
    tl.store(Ages + s, tl.where(used, tick, age), s < SLOTS)
    tl.store(CopyExperts + s, copy_expert, s < SLOTS)
    tl.store(Clock, tick)

    # Compact active experts into a bounded GPU group list. Decode launches at
    # most top_k groups; prefill can launch all local experts without host reads.
    group = tl.cumsum(active.to(tl.int32)) - 1
    tl.store(Experts + e, -1, e < LOCAL)
    tl.store(Counts + e, 0, e < LOCAL)
    tl.debug_barrier()
    tl.store(Experts + group, START + e, active)
    tl.store(Counts + group, counts, active)
    positions = tl.cumsum(matches.to(tl.int32), 1) - 1
    tl.store(Rows + group[:, None] * CAPACITY + positions, r[None, :], matches)

    # loop that updates the pointer table
    for bank in tl.static_range(STRIDE):
        source = tl.load(Sources + e * STRIDE + bank, e < LOCAL, 0)
        destination = tl.load(
            Destinations + slot * STRIDE + bank, (e < LOCAL) & (slot >= 0), 0
        )
        # Overflow prefill experts read mapped host weights directly. Decode
        # fits in the cache, so all its GEMMs read GPU storage.
        pointer = tl.where(~resident & (slot >= 0), destination, source)
        tl.store(
            Pointers + (START + e) * STRIDE + bank,
            tl.where(active, pointer, 0),
            e < LOCAL,
        )


@triton.jit
def _copy(
    Sources,
    Destinations,
    CopyExperts,
    SIZES: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    WORKERS: tl.constexpr,
):
    worker = tl.program_id(0)
    slot = tl.program_id(1)
    expert = tl.load(CopyExperts + slot)
    if expert >= 0:
        for bank in tl.static_range(STRIDE):
            source = tl.load(Sources + expert * STRIDE + bank).to(
                tl.pointer_type(tl.int32)
            )
            target = tl.load(Destinations + slot * STRIDE + bank).to(
                tl.pointer_type(tl.int32)
            )
            for tile in range(worker, tl.cdiv(SIZES[bank], BLOCK), WORKERS):
                offsets = tile * BLOCK + tl.arange(0, BLOCK)
                value = tl.load(source + offsets, offsets < SIZES[bank], 0)
                tl.store(target + offsets, value, offsets < SIZES[bank])


class ExpertCache:
    """
    Contiguous EP shard; fixed GPU residents, LRU slots, sequential forwards.
    """

    def __init__(self, weights, names, num_slots, grouped):
        self.weights = weights  # Keep mapped host allocations alive.
        ids = sorted(weights[names[0]])
        self.start = ids[0]
        self.local = len(ids)
        assert ids == list(range(self.start, self.start + self.local))
        assert num_slots > 0
        self.num_slots = num_slots
        self.stride = len(names)
        device = grouped.device
        self.banks = [
            torch.empty(
                (num_slots, *weights[name][ids[0]].shape),
                dtype=weights[name][ids[0]].dtype,
                device=device,
            )
            for name in names
        ]
        self.sources = torch.tensor(
            [[mapped_pointer(weights[name][e]) for name in names] for e in ids],
            dtype=torch.int64,
            device=device,
        )
        self.destinations = torch.tensor(
            [[bank[s].data_ptr() for bank in self.banks] for s in range(num_slots)],
            dtype=torch.int64,
            device=device,
        )
        self.resident = torch.tensor(
            [weights[names[0]][e].is_cuda for e in ids],
            dtype=torch.int32,
            device=device,
        )
        byte_sizes = [
            weights[name][ids[0]].numel() * weights[name][ids[0]].element_size()
            for name in names
        ]
        assert all(size % 4 == 0 for size in byte_sizes)
        self.sizes = tuple(size // 4 for size in byte_sizes)
        self.slot_experts = torch.full(
            (num_slots,), -1, dtype=torch.int32, device=device
        )
        self.ages = torch.zeros(num_slots, dtype=torch.int32, device=device)
        self.clock = torch.zeros((), dtype=torch.int32, device=device)
        self.copy_experts = torch.empty(num_slots, dtype=torch.int32, device=device)
        self.grouped = grouped

    def prepare(self, indices):
        routes = indices.numel()
        self.grouped.reserve(routes)
        _plan[(1,)](
            indices,
            self.sources,
            self.resident,
            self.destinations,
            self.slot_experts,
            self.ages,
            self.clock,
            self.copy_experts,
            self.grouped.pointers,
            self.grouped.experts,
            self.grouped.counts,
            self.grouped.rows,
            self.start,
            self.local,
            self.num_slots,
            self.stride,
            routes,
            self.grouped.capacity,
            triton.next_power_of_2(self.local),
            triton.next_power_of_2(self.num_slots),
            triton.next_power_of_2(routes),
            num_warps=4,
        )
        _copy[(128, self.num_slots)](
            self.sources,
            self.destinations,
            self.copy_experts,
            self.sizes,
            self.stride,
            1024,
            128,
        )
        self.grouped.group_size = min(routes, self.local)
