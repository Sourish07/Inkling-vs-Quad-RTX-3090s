"""Small, offline CPU/fp32 comparisons with Transformers' Inkling modules.

The vision-tower case is temporarily skipped until that tower is implemented.
Run with ``uv run --frozen -m pytest tests/test_model_parity.py``.
"""

from collections.abc import Iterator

import pytest
import torch
from torch import nn
from transformers.models.inkling import modeling_inkling as hf
from transformers.models.inkling.configuration_inkling import (
    InklingAudioConfig,
    InklingConfig,
    InklingTextConfig,
    InklingVisionConfig,
)

from my_inkling import MyInkling
from my_inkling import model as inkling
from my_inkling.cache import MyInklingCache


@pytest.fixture(autouse=True)
def cpu_fp32() -> Iterator[None]:
    """Keep tiny ops fast and restore the caller's RNG, dtype and thread settings."""
    previous_dtype = torch.get_default_dtype()
    previous_threads = torch.get_num_threads()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(0)
            yield
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


@pytest.fixture
def text_config() -> InklingTextConfig:
    config = InklingTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=4,
        sliding_window_size=3,
        d_rel=2,
        rel_extent=4,
        max_position_embeddings=32,
        conv_kernel_size=3,
        layer_types=["hybrid", "hybrid_sliding"],
        mlp_layer_types=["dense", "sparse"],
        intermediate_size=24,
        moe_intermediate_size=12,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=2,
        route_scale=1.7,
        logits_mup_width_multiplier=2.0,
        log_scaling_n_floor=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    # Only eager attention consumes Inkling's relative position bias.
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


@pytest.fixture
def config(text_config: InklingTextConfig) -> InklingConfig:
    return InklingConfig(
        text_config=text_config,
        audio_config=InklingAudioConfig(n_mel_bins=2, mel_vocab_size=4),
        vision_config=InklingVisionConfig(
            patch_size=2,
            temporal_patch_size=2,
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
        ),
        image_token_id=30,
        audio_token_id=31,
        image_bos_token_id=28,
        audio_bos_token_id=29,
    )


def pair[ActualModule: nn.Module, ReferenceModule: nn.Module](
    actual: ActualModule,
    reference: ReferenceModule,
    ignored_prefixes: tuple[str, ...] = (),
) -> tuple[ActualModule, ReferenceModule]:
    """Initialize even torch.empty expert tensors, then copy identical weights."""
    actual = actual.to(device="cpu", dtype=torch.float32).eval()
    reference = reference.to(device="cpu", dtype=torch.float32).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith("global_scale"):
                parameter.fill_(1.3)
            elif parameter.ndim == 1:
                parameter.uniform_(0.7, 1.3)
            else:
                parameter.normal_(mean=0.0, std=0.2)
        for module in reference.modules():
            if isinstance(module, hf.InklingTopkRouter):
                module.e_score_correction_bias.copy_(
                    torch.linspace(-0.3, 0.3, module.num_experts)
                )
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()

    actual_state = actual.state_dict()
    reference_state = reference.state_dict()
    actual_keys = {key for key in actual_state if not key.startswith(ignored_prefixes)}
    reference_keys = {
        key for key in reference_state if not key.startswith(ignored_prefixes)
    }
    assert actual_keys == reference_keys, (
        f"{type(actual).__name__}: missing weights={reference_keys - actual_keys}, "
        f"unexpected weights={actual_keys - reference_keys}"
    )
    weights = {key: reference_state[key].clone() for key in reference_keys}
    loaded = actual.load_state_dict(weights, strict=not ignored_prefixes)
    assert not loaded.unexpected_keys
    assert all(key.startswith(ignored_prefixes) for key in loaded.missing_keys)
    for module in (actual, reference):
        for tensor in module.state_dict().values():
            assert tensor.device.type == "cpu"
            if tensor.is_floating_point():
                assert tensor.dtype == torch.float32
    for module in actual.modules():
        if isinstance(module, inkling.MyInklingAttention):
            module.fuse_projections()
    return actual, reference


def assert_fp32_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == expected.device.type == "cpu"
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def hidden_states(config: InklingTextConfig) -> torch.Tensor:
    return torch.randn(2, 7, config.hidden_size)


def token_ids() -> torch.Tensor:
    # Unpadded sequences longer than both convolution and sliding windows.
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])


def attention_mask(config: InklingTextConfig, layer_idx: int) -> torch.Tensor:
    positions = torch.arange(token_ids().shape[1])
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0)[None, None]
    if config.layer_types[layer_idx] == "hybrid_sliding":
        allowed = allowed & (distance < config.sliding_window_size)[None, None]
    return torch.zeros(allowed.shape).masked_fill(
        ~allowed, torch.finfo(torch.float32).min
    )


@torch.no_grad()
@pytest.mark.parametrize("query_offset", [0, 5])
def test_relative_logits_fp32(
    query_offset: int, text_config: InklingTextConfig
) -> None:
    actual, reference = pair(
        inkling.MyInklingRelativeLogits(text_config.d_rel, text_config.rel_extent),
        hf.InklingRelativeLogits(text_config.d_rel, text_config.rel_extent),
    )
    states = torch.randn(2, 7, 4, text_config.d_rel)
    queries = torch.arange(7) + query_offset
    keys = torch.arange(12)
    expected = reference(states, queries, keys)
    assert_fp32_close(actual(states, queries[:, None] - keys[None, :]), expected)


@torch.no_grad()
def test_rms_norm_fp32(text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingRMSNorm(text_config.hidden_size, text_config.rms_norm_eps),
        hf.InklingRMSNorm(text_config.hidden_size, text_config.rms_norm_eps),
    )
    states = hidden_states(text_config)
    states[0, 0].zero_()
    states[0, 1].mul_(1e-8)
    expected = reference(states)
    assert_fp32_close(actual(states), expected)


@torch.no_grad()
def test_mlp_fp32(text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingMLP(text_config), hf.InklingMLP(text_config)
    )
    states = hidden_states(text_config)
    expected = reference(states)
    assert_fp32_close(actual(states), expected)


@torch.no_grad()
def test_shared_experts_fp32(text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingSharedExperts(text_config),
        hf.InklingSharedExperts(text_config),
    )
    states = hidden_states(text_config)
    gammas = torch.rand(states.shape[0] * states.shape[1], text_config.n_shared_experts)
    gammas[0].zero_()
    expected = reference(states, gammas)
    assert_fp32_close(actual(states, gammas), expected)


@torch.no_grad()
def test_experts_fp32(text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingExperts(text_config), hf.InklingExperts(text_config)
    )
    states = torch.randn(5, text_config.hidden_size)
    # Repeated experts, unequal weights and an unused expert exercise dispatch/sum.
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]])
    weights = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.0, 1.0], [0.6, 0.4]])
    expected = reference(states, indices, weights)
    assert_fp32_close(actual(states, indices, weights), expected)


@torch.no_grad()
def test_topk_router_fp32(text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingTopkRouter(text_config), hf.InklingTopkRouter(text_config)
    )
    states = hidden_states(text_config)
    expected_logits, expected_weights, expected_indices, expected_gammas = reference(
        states
    )
    logits, weights, indices, gammas = actual(states)
    assert_fp32_close(logits, expected_logits)
    assert_fp32_close(gammas, expected_gammas)
    # topk(sorted=False) may return the same experts in a different order.
    expected_indices, expected_order = expected_indices.sort(dim=-1)
    indices, order = indices.sort(dim=-1)
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    assert_fp32_close(
        weights.gather(-1, order), expected_weights.gather(-1, expected_order)
    )


@torch.no_grad()
def test_moe_fp32(text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingMoE(text_config), hf.InklingMoE(text_config)
    )
    states = hidden_states(text_config)
    expected = reference(states)
    assert_fp32_close(actual(states), expected)


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [1, 7])
def test_short_conv_fp32(seq_len: int, text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingShortConv(text_config.hidden_size, 3, layer_idx=0, conv_idx=0),
        hf.InklingShortConvolution(text_config.hidden_size, 3, layer_idx=0, conv_idx=0),
    )
    states = hidden_states(text_config)[:, :seq_len]
    expected = reference(states)
    assert_fp32_close(actual(states), expected)


@torch.no_grad()
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["full", "sliding"])
def test_attention_fp32(layer_idx: int, text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingAttention(text_config, layer_idx),
        hf.InklingAttention(text_config, layer_idx),
    )
    states = hidden_states(text_config)
    mask = attention_mask(text_config, layer_idx)
    expected, _ = reference(states, attention_mask=mask)
    assert_fp32_close(actual(states), expected)


@torch.no_grad()
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["full", "sliding"])
def test_attention_cached_fp32(layer_idx: int, text_config: InklingTextConfig) -> None:
    """Decode through SWA rollover and past the learned relative-bias extent."""
    actual, reference = pair(
        inkling.MyInklingAttention(text_config, layer_idx),
        hf.InklingAttention(text_config, layer_idx),
    )
    states = torch.randn(1, 10, text_config.hidden_size)
    positions = torch.arange(states.shape[1])
    distance = positions[:, None] - positions[None, :]
    allowed = distance >= 0
    if layer_idx == 1:
        allowed &= distance < text_config.sliding_window_size
    mask = torch.zeros(1, 1, 10, 10).masked_fill(~allowed, float("-inf"))
    expected, _ = reference(states, attention_mask=mask)
    cache = MyInklingCache(text_config)
    chunks = [2] + [1] * 8
    start = 0
    for size in chunks:
        end = start + size
        assert_fp32_close(actual(states[:, start:end], cache), expected[:, start:end])
        start = end


@torch.no_grad()
def test_normed_embedding_fp32(text_config: InklingTextConfig) -> None:
    args = (
        text_config.vocab_size,
        text_config.hidden_size,
        text_config.pad_token_id,
        text_config.rms_norm_eps,
    )
    actual, reference = pair(
        inkling.MyInklingNormedEmbedding(*args), hf.InklingNormedEmbedding(*args)
    )
    ids = token_ids()
    expected = reference(ids)
    assert_fp32_close(actual(ids), expected)


@torch.no_grad()
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["full", "sliding"])
@pytest.mark.parametrize("mlp_type", ["dense", "sparse"])
def test_decoder_layer_fp32(
    layer_idx: int, mlp_type: str, text_config: InklingTextConfig
) -> None:
    text_config.mlp_layer_types[layer_idx] = mlp_type
    actual, reference = pair(
        inkling.MyInklingDecoderLayer(text_config, layer_idx),
        hf.InklingDecoderLayer(text_config, layer_idx),
    )
    states = hidden_states(text_config)
    mask = attention_mask(text_config, layer_idx)
    expected = reference(states, attention_mask=mask)
    assert_fp32_close(actual(states), expected)


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [1, 7])
def test_text_tower_fp32(seq_len: int, text_config: InklingTextConfig) -> None:
    actual, reference = pair(
        inkling.MyInklingTextTower(text_config), hf.InklingTextModel(text_config)
    )
    ids = token_ids()[:, :seq_len]
    expected = reference(input_ids=ids, use_cache=False).last_hidden_state
    output = actual(ids)
    assert_fp32_close(output, expected)


@torch.no_grad()
@pytest.mark.parametrize(
    "with_lm_head", [False, True], ids=["model", "conditional_generation"]
)
def test_model_fp32(with_lm_head: bool, config: InklingConfig) -> None:
    if with_lm_head:
        actual = MyInkling(config)
        reference = hf.InklingForConditionalGeneration(config)
        ignored = ("model.audio_tower.", "model.vision_tower.")
        output_field = "logits"
    else:
        actual = inkling.MyInklingModel(config)
        reference = hf.InklingModel(config)
        ignored = ("audio_tower.", "vision_tower.")
        output_field = "last_hidden_state"
    # Text-only inputs do not use audio/vision weights. Vision has its own test;
    # model.py does not yet define an audio module.
    actual, reference = pair(actual, reference, ignored_prefixes=ignored)
    ids = token_ids()
    expected = getattr(reference(input_ids=ids, use_cache=False), output_field)
    if with_lm_head:
        expected = expected[:, -1:, :]
    assert_fp32_close(actual(ids), expected)


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [1, 7])
def test_last_token_logits_fp32(seq_len: int, config: InklingConfig) -> None:
    config.text_config.unpadded_vocab_size = 29
    actual, reference = pair(
        MyInkling(config),
        hf.InklingForConditionalGeneration(config),
        ignored_prefixes=("model.audio_tower.", "model.vision_tower."),
    )
    ids = token_ids()[:, :seq_len]
    expected = reference(input_ids=ids, use_cache=False).logits[:, -1:, :]
    output = actual(ids)
    assert output.shape == (ids.shape[0], 1, 29)
    assert_fp32_close(output, expected)
