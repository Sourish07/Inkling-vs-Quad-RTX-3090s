"""Pure-PyTorch NVFP4 quantization and Marlin weight preparation.

Port of sglang's ``prepare_nvfp4_layer_for_marlin``
(``srt/layers/quantization/marlin_utils_fp4.py``) with the CUDA
``gptq_marlin_repack`` kernel replaced by the pure-torch permutation it
implements. ``sglang_ref.check_pack_matches_sglang`` asserts the two are
bit-identical.

Layout pipeline (offline, perf-irrelevant):

    w [N, K] float
      -> nvfp4_quantize: e2m1 codes [N, K] (4-bit), one fp8-e4m3 block scale per
         16 consecutive K elements [N, K/16], one global scale
      -> marlin_pack_weight: codes.T [K, N] cut into 16x16 tiles, four at a time
         reordered into the mma.m16n8k16 B-fragment order (get_weight_perm),
         8 nibbles/int32
      -> marlin_pack_scale: scales.T [K/16, N], 8x8 transpose per 64 columns,
         pair interleave, then re-encoded as the top byte of fp16(s * 2^7) << 1
      -> marlin_pack_scale2: g * 2^(exponent_bias - 7) in the activation dtype

The kernel-facing binding is ``nvfp4_linear(x, weight, scale, scale2)``:

    x       [M, K]       fp16 | bf16
    weight  [K/16, 2*N]  int32            (marlin_pack_weight)
    scale   [K/16, N]    float8_e4m3fn    (marlin_pack_scale)
    scale2  [1]          same as x        (marlin_pack_scale2)
    out     [M, N]       same as x

with N % 128 == 0 and K % 64 == 0.

What the kernel computes, bit for bit (see dequant.h in sglang):

  * weight nibble [s e1 e0 m] is OR-ed straight into the activation dtype's
    sign / low-exponent / top-mantissa bits ("dequant_skip_flop"), giving
    e2m1(code) * 2^-14 for fp16 and e2m1(code) * 2^-126 for bf16. The e2m1
    subnormal (0.5) lands on a subnormal of the wide type, so it stays exact.
  * scale byte b is placed at fp16 bits 14..7 (``b << 7``), i.e. the value
    s * 2^7. For bf16 the byte's top bit becomes the exponent MSB and the low
    7 bits go to bits 10..4 -- the same value s * 2^7 for every *normal* e4m3
    scale (top bit set), and ~0 for subnormal ones (a quirk kept as-is).
  * B = weight * scale (packed fp16/bf16 multiply), C = A @ B with fp32
    accumulation, then C -> activation dtype and multiplied by scale2.

``decode_binding`` rebuilds exactly those bit patterns, so
``reference_linear`` is an fp64 oracle for any (weight, scale, scale2).
"""

from dataclasses import dataclass

import torch

MARLIN_TILE = 16
GROUP_SIZE = 16

# e2m1 magnitudes indexed by the low three bits [e1 e0 m]; bit 3 is the sign.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
E4M3_MAX = 448.0

# exponent_bias = 2^(target_exponent_bits - 1) - 2^(fp4_exponent_bits - 1)
_EXPONENT_BIAS = {torch.float16: 14, torch.bfloat16: 126}


# ---------------------------------------------------------------------------
# Permutations (vLLM marlin_perms.py / sglang gptq_marlin_repack.cuh, 4-bit)
# ---------------------------------------------------------------------------


def get_weight_perm() -> torch.Tensor:
    """Gather indices taking four 16x16 tiles (64 columns) to packed order.

    The source chunk is four consecutive [16 (k), 16 (n)] tiles, each
    row-major, so element (k, n) sits at ``256 * (n // 16) + 16 * k + n % 16``.
    Position p of the result holds element ``perm[p]`` of that chunk. The
    stream is ordered by lane (32) -> n16-block j (4) -> 8 nibbles, i.e. each
    lane owns one int4 (4 words) per k16 row; lane i owns column ``i // 4``
    and k rows ``{2r, 2r+1, 2r+8, 2r+9}`` with ``r = i % 4``, which is the
    mma.m16n8k16 B fragment. The 8 nibbles of a word cover both n8 halves of
    the n16 block and are interleaved [0,2,4,6,1,3,5,7] so one mask+shift
    extracts two values at a time.
    """
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    perm = torch.tensor(perm_list, dtype=torch.long)
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    return perm.reshape(-1, 8)[:, interleave].reshape(-1)


def get_scale_perm() -> torch.Tensor:
    """Grouped-scale permutation over 64 columns: an 8x8 transpose."""
    return torch.tensor([i + 8 * j for i in range(8) for j in range(8)])


_PAIR_INTERLEAVE = (0, 2, 1, 3)


# ---------------------------------------------------------------------------
# e2m1 / NVFP4 quantization
# ---------------------------------------------------------------------------


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    """4-bit e2m1 codes (uint8, 0..15) -> float64 values."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float64, device=codes.device)
    mag = table[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value; returns uint8 codes (0..15)."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=x.device)
    mag = x.abs().float().clamp(max=E2M1_MAX)
    code = (mag[..., None] - table).abs().argmin(dim=-1).to(torch.uint8)
    return code | ((x < 0).to(torch.uint8) << 3)


@dataclass
class Nvfp4Weight:
    """An NVFP4 weight in ModelOpt's on-disk layout."""

    codes: torch.Tensor  # [N, K] uint8, one e2m1 code per element
    block_scale: torch.Tensor  # [N, K/16] float8_e4m3fn
    global_scale: torch.Tensor  # [] float32, multiplicative

    @property
    def packed(self) -> torch.Tensor:
        """[N, K/2] uint8, two nibbles per byte (even element in the low nibble)."""
        return self.codes[:, 0::2] | (self.codes[:, 1::2] << 4)

    def dequantize(self) -> torch.Tensor:
        """Real-valued weight [N, K] in float64."""
        s = self.block_scale.double().repeat_interleave(GROUP_SIZE, dim=1)
        return e2m1_decode(self.codes) * s * self.global_scale.double()


def nvfp4_quantize(w: torch.Tensor) -> Nvfp4Weight:
    """ModelOpt-style NVFP4 quantization of a float weight [N, K]."""
    size_n, size_k = w.shape
    assert size_k % GROUP_SIZE == 0
    blocks = w.float().reshape(size_n, size_k // GROUP_SIZE, GROUP_SIZE)
    amax = blocks.abs().amax(dim=-1)
    global_scale = (amax.max() / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)
    block_scale = (amax / (E2M1_MAX * global_scale)).to(torch.float8_e4m3fn)
    eff = (block_scale.float() * global_scale).clamp(min=1e-30)
    codes = e2m1_encode(blocks / eff[..., None]).reshape(size_n, size_k)
    return Nvfp4Weight(codes, block_scale, global_scale)


# ---------------------------------------------------------------------------
# Marlin packing (forward and inverse)
# ---------------------------------------------------------------------------


def marlin_pack_weight(codes: torch.Tensor) -> torch.Tensor:
    """e2m1 codes [N, K] uint8 -> Marlin weight [K/16, 2*N] int32."""
    size_n, size_k = codes.shape
    assert size_k % MARLIN_TILE == 0 and size_n % 64 == 0
    perm = get_weight_perm().to(codes.device)
    q = codes.T.reshape(
        size_k // MARLIN_TILE, MARLIN_TILE, size_n // MARLIN_TILE, MARLIN_TILE
    )
    q = q.permute(0, 2, 1, 3).reshape(-1, perm.numel())[:, perm]
    q = q.reshape(size_k // MARLIN_TILE, size_n * 2, 8).to(torch.int64)
    shifts = 4 * torch.arange(8, device=codes.device)
    packed = (q << shifts).sum(dim=-1)
    # Reinterpret the unsigned 32-bit word as int32.
    return (((packed + 2**31) % 2**32) - 2**31).to(torch.int32).contiguous()


def marlin_unpack_weight(weight: torch.Tensor) -> torch.Tensor:
    """Inverse of marlin_pack_weight: [K/16, 2*N] int32 -> codes [N, K] uint8."""
    k_tiles, two_n = weight.shape
    size_k, size_n = k_tiles * MARLIN_TILE, two_n // 2
    perm = get_weight_perm().to(weight.device)
    shifts = 4 * torch.arange(8, device=weight.device)
    q = ((weight.to(torch.int64)[..., None] >> shifts) & 0xF).to(torch.uint8)
    q = q.reshape(-1, perm.numel())
    tile = torch.empty_like(q)
    tile[:, perm] = q
    tile = tile.reshape(k_tiles, size_n // MARLIN_TILE, MARLIN_TILE, MARLIN_TILE)
    tile = tile.permute(0, 2, 1, 3)
    return tile.reshape(size_k, size_n).T.contiguous()


def marlin_pack_scale(block_scale: torch.Tensor) -> torch.Tensor:
    """fp8 block scales [N, K/16] -> Marlin scale [K/16, N] float8_e4m3fn.

    marlin_permute_scales(group_size=16) + nvfp4_marlin_process_scales. The
    stored byte is NOT an e4m3 encoding of the scale: it is bits 14..7 of
    fp16(s * 2^7), viewed through the e4m3 dtype.
    """
    size_n = block_scale.shape[0]
    assert size_n % 64 == 0
    s = block_scale.T.contiguous().to(torch.float16)
    s = s.reshape(-1, 64)[:, get_scale_perm().to(s.device)].reshape(-1, size_n)
    s = s.reshape(-1, 4)[:, list(_PAIR_INTERLEAVE)].reshape(s.size(0), -1)
    s = (s * (2**7)).view(torch.int16) << 1
    return s.view(torch.float8_e4m3fn)[:, 1::2].contiguous()


def marlin_unpack_scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    """Marlin scale [K/16, N] -> raw scale bytes in natural order [N, K/16]."""
    groups, size_n = scale.shape
    b = scale.view(torch.uint8)
    b = b.reshape(-1, 4)[:, list(_PAIR_INTERLEAVE)].reshape(groups, size_n)
    out = torch.empty_like(b).reshape(-1, 64)
    out[:, get_scale_perm().to(b.device)] = b.reshape(-1, 64)
    return out.reshape(groups, size_n).T.contiguous()


def marlin_pack_scale2(global_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Global scale -> [1] tensor in the activation dtype, bias-corrected."""
    g = global_scale.to(dtype).reshape(1)
    return g * (2.0 ** (_EXPONENT_BIAS[dtype] - 7))


@dataclass
class MarlinNvfp4:
    """The three weight-side tensors of the nvfp4_linear binding."""

    weight: torch.Tensor  # [K/16, 2*N] int32
    scale: torch.Tensor  # [K/16, N] float8_e4m3fn
    scale2: torch.Tensor  # [1] activation dtype


def marlin_prepare(q: Nvfp4Weight, dtype: torch.dtype) -> MarlinNvfp4:
    return MarlinNvfp4(
        weight=marlin_pack_weight(q.codes),
        scale=marlin_pack_scale(q.block_scale),
        scale2=marlin_pack_scale2(q.global_scale, dtype),
    )


# ---------------------------------------------------------------------------
# Bit-level oracle
# ---------------------------------------------------------------------------


def _to_i16(bits: torch.Tensor) -> torch.Tensor:
    """Reinterpret unsigned 16-bit patterns held in a wider int as int16."""
    return (((bits.to(torch.int32) + 2**15) % 2**16) - 2**15).to(torch.int16)


def decode_binding(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantized B [K, N] in float64, in the kernel's scaled domain.

    Rebuilds the exact bit patterns the kernel's dequant produces (see the
    module docstring). The result equals ``w * s * 2^(7 - exponent_bias)``.
    """
    codes = marlin_unpack_weight(weight).to(torch.int32)  # [N, K]
    sbytes = marlin_unpack_scale_bytes(scale).to(torch.int32)  # [N, K/16]
    if dtype == torch.float16:
        w_bits = ((codes & 8) << 12) | ((codes & 7) << 9)
        s_bits = sbytes << 7
    elif dtype == torch.bfloat16:
        w_bits = ((codes & 8) << 12) | ((codes & 7) << 6)
        s_bits = ((sbytes & 0x80) << 7) | ((sbytes & 0x7F) << 4)
    else:
        raise ValueError(f"unsupported activation dtype {dtype}")
    w = _to_i16(w_bits).view(dtype).double()
    s = _to_i16(s_bits).view(dtype).double()
    return (w * s.repeat_interleave(GROUP_SIZE, dim=1)).T.contiguous()


def reference_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor
) -> torch.Tensor:
    """fp64 oracle for nvfp4_linear: [M, N] float64, no output rounding."""
    b = decode_binding(weight, scale, x.dtype)
    return (x.double() @ b) * scale2.double()


def random_binding(
    size_k: int,
    size_n: int,
    dtype: torch.dtype,
    *,
    seed: int = 0,
    realistic: bool = True,
    device: str = "cuda",
) -> tuple[MarlinNvfp4, Nvfp4Weight]:
    """A random NVFP4 layer, already Marlin-packed.

    ``realistic=True`` quantizes a Gaussian weight (block scales cluster near
    the top of the e4m3 range). ``realistic=False`` draws every nibble
    uniformly (all 16 codes, including -0 and the +-0.5 subnormal) with block
    scales spread over ~10 binades of *normal* e4m3 values -- a bit-pattern
    stress test rather than a plausible layer.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if realistic:
        w = torch.randn(size_n, size_k, generator=gen) * 0.02
        q = nvfp4_quantize(w.to(device))
    else:
        codes = torch.randint(0, 16, (size_n, size_k), generator=gen, dtype=torch.uint8)
        # e4m3 bytes with exponent field 4..14 are normal and avoid NaN (0x7f).
        exp = torch.randint(4, 15, (size_n, size_k // GROUP_SIZE), generator=gen)
        man = torch.randint(0, 8, (size_n, size_k // GROUP_SIZE), generator=gen)
        byte = ((exp << 3) | man).to(torch.uint8)
        q = Nvfp4Weight(
            codes.to(device),
            byte.view(torch.float8_e4m3fn).to(device),
            torch.tensor(2.0**-9, device=device),
        )
    return marlin_prepare(q, dtype), q
