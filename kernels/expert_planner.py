"""
GPU cache planning and onload from CUDA-mapped pinned checkpoint tensors.
"""

import triton
import triton.language as tl


@triton.jit
def expert_plan(
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
    """
    Groups token routing by expert, assign LRU slots to cache misses, write to the pointer table.

    Runs as one program so all decisions see the same cache state. Planning on
    the GPU means the host never downloads (from GPU) expert IDs (a forced sync) and never
    rebuilds or uploads the pointer table.

    Params:
        Clock: just an int that gets incremented every time the kernel runs, used to update Ages
    """
    # Step 1: count number of tokens routed to each expert; mark active experts
    # we're simply consuming the output of TopKRouter
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

    # Step 2: LRU; assign slots to cache misses
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

    # Step 3: Compact active experts into a bounded GPU group list
    # Decode launches at most top_k groups; prefill can launch all local experts without host reads.
    # used to be _route previous, logic is ported here (stream compaction)
    group = tl.cumsum(active.to(tl.int32)) - 1
    tl.store(Experts + e, -1, e < LOCAL)
    tl.store(Counts + e, 0, e < LOCAL)
    tl.debug_barrier()
    tl.store(Experts + group, START + e, active)
    tl.store(Counts + group, counts, active)
    positions = tl.cumsum(matches.to(tl.int32), 1) - 1
    tl.store(Rows + group[:, None] * CAPACITY + positions, r[None, :], matches)

    # Step 4: loop that updates the pointer table
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
def expert_copy(
    Sources,
    Destinations,
    CopyExperts,
    SIZES: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    WORKERS: tl.constexpr,
):
    """
    Copy the expert that `expert_plan` assigned to this slot from host memory into
    the slot.

    Only the GPU knows which experts missed the cache, so the host cannot start
    these copies. Each program reads the pinned host weights directly; WORKERS
    programs per slot share the tiles. Slots with no assignment exit at once.
    """
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
