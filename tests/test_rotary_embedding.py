# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared RotaryEmbedding and RoPE helpers."""

import pytest
import torch

from vllm_rlt.layers import (
    RotaryEmbedding,
    apply_interleaved_rotary_pos_emb,
    apply_rotary_pos_emb,
    rotate_half,
)
from vllm_rlt.models.nanbeige import (
    NanbeigeRotaryEmbedding,
)
from vllm_rlt.models.nanbeige import (
    _rotate_half as nanbeige_rotate_half,
)
from vllm_rlt.models.ouro import (
    OuroRotaryEmbedding,
)
from vllm_rlt.models.ouro import (
    _rotate_half as ouro_rotate_half,
)


def test_backward_compatible_aliases():
    assert OuroRotaryEmbedding is RotaryEmbedding
    assert NanbeigeRotaryEmbedding is RotaryEmbedding
    assert ouro_rotate_half is rotate_half
    assert nanbeige_rotate_half is rotate_half


def test_rotate_half_mathematics():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotated = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    torch.testing.assert_close(rotated, expected)


def test_apply_rotary_pos_emb():
    b, heads, dim = 2, 4, 8
    q = torch.randn(b, heads, dim)
    k = torch.randn(b, heads, dim)
    cos = torch.randn(b, 1, dim)
    sin = torch.randn(b, 1, dim)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    expected_q = q * cos + rotate_half(q) * sin
    expected_k = k * cos + rotate_half(k) * sin

    torch.testing.assert_close(q_rot, expected_q)
    torch.testing.assert_close(k_rot, expected_k)


def test_rotary_embedding_initialization_and_direct_arguments():
    head_dim = 64
    rope_theta = 500000.0
    rotary = RotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta)

    assert rotary.head_dim == head_dim
    assert rotary.rope_theta == rope_theta
    assert rotary.inv_freq.shape == (head_dim // 2,)
    assert rotary.inv_freq.dtype == torch.float32

    expected_freqs = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    torch.testing.assert_close(rotary.inv_freq, expected_freqs)


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.bfloat16])
def test_rotary_embedding_dtype_conversion_preserves_float32(target_dtype):
    rotary = RotaryEmbedding(head_dim=32, rope_theta=10000.0)
    expected_freqs = rotary.inv_freq.clone()

    rotary.to(target_dtype)
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, expected_freqs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_embedding_forward_precision_and_shapes(dtype):
    head_dim = 16
    rotary = RotaryEmbedding(head_dim=head_dim, rope_theta=10000.0).to(dtype)
    positions = torch.tensor([0, 1, 5, 10])
    hidden = torch.randn(len(positions), 32, dtype=dtype)

    cos, sin = rotary(hidden, positions)
    assert cos.shape == (len(positions), 1, head_dim)
    assert sin.shape == (len(positions), 1, head_dim)
    assert cos.dtype == dtype
    assert sin.dtype == dtype


def _interleaved_complex_reference(q, k, freqs):
    rotation = torch.view_as_complex(freqs.double().contiguous())
    return tuple(
        torch.view_as_real(
            torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2).contiguous()) * rotation
        ).flatten(-2)
        for x in (q, k)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_interleaved_rotary_matches_complex_rotation(dtype):
    rows, heads, dim = 5, 3, 8
    q, k = (torch.randn(rows, heads, dim).to(dtype) for _ in range(2))
    angles = torch.randn(rows, 1, dim // 2)
    freqs = torch.stack((angles.cos(), angles.sin()), dim=-1)

    q_rot, k_rot = apply_interleaved_rotary_pos_emb(q, k, freqs)
    expected_q, expected_k = _interleaved_complex_reference(q, k, freqs)

    assert q_rot.dtype == k_rot.dtype == dtype
    tolerance = {"rtol": 1e-6, "atol": 1e-6} if dtype == torch.float32 else {}
    torch.testing.assert_close(q_rot, expected_q.to(dtype), **tolerance)
    torch.testing.assert_close(k_rot, expected_k.to(dtype), **tolerance)


# Default (query heads, key heads, head dim) of OuroConfig, NanbeigeConfig and DoryConfig.
SPLIT_HALF_LAYOUTS = [(16, 16, 128), (48, 8, 128), (16, 4, 256)]
# Default (heads, head dim) of HuginnConfig.
INTERLEAVED_LAYOUT = (55, 96)


def _split_half_factors(rows, dim, dtype, theta=10000.0):
    rotary = RotaryEmbedding(head_dim=dim, rope_theta=theta).cuda()
    positions = torch.randint(0, 32768, (rows,), device="cuda")
    return rotary(torch.empty(rows, 1, dtype=dtype, device="cuda"), positions)


def _split_half_exact(q, k, cos, sin):
    cos, sin = cos.double(), sin.double()
    return tuple(x.double() * cos + rotate_half(x.double()) * sin for x in (q, k))


def _rounding_tolerance(dtype):
    # One unit in the last place of the output dtype, plus FP32 arithmetic error.
    if dtype == torch.float32:
        return {"rtol": 1e-6, "atol": 1e-5}
    return {"rtol": torch.finfo(dtype).eps, "atol": 1e-5}


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", SPLIT_HALF_LAYOUTS)
@pytest.mark.parametrize("rows", [1, 7, 512])
def test_fused_split_half_rope_rounds_once(dtype, layout, rows):
    query_heads, key_heads, dim = layout
    q = torch.randn(rows, query_heads, dim, device="cuda").to(dtype)
    k = torch.randn(rows, key_heads, dim, device="cuda").to(dtype)
    cos, sin = _split_half_factors(rows, dim, dtype)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    torch_q = q * cos + rotate_half(q) * sin
    torch_k = k * cos + rotate_half(k) * sin

    for fused, eager, exact in zip(
        (q_rot, k_rot), (torch_q, torch_k), _split_half_exact(q, k, cos, sin), strict=True
    ):
        assert fused.dtype == dtype and fused.shape == exact.shape
        torch.testing.assert_close(fused.double(), exact, **_rounding_tolerance(dtype))
        if dtype != torch.float32:  # FP32 eager already computes in the output dtype
            assert (fused.double() - exact).abs().max() <= (eager.double() - exact).abs().max()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_split_half_rope_reads_strided_projection_views(dtype):
    rows, query_heads, key_heads, dim = 9, 16, 4, 64
    widths = (query_heads * dim, key_heads * dim, key_heads * dim)
    packed = torch.randn(rows, sum(widths), device="cuda").to(dtype)
    q, k, _ = (part.view(rows, -1, dim) for part in packed.split(widths, dim=-1))
    assert not q.is_contiguous() and not k.is_contiguous()
    cos, sin = _split_half_factors(rows, dim, dtype)

    fused = apply_rotary_pos_emb(q, k, cos, sin)
    contiguous = apply_rotary_pos_emb(q.contiguous(), k.contiguous(), cos, sin)

    for actual, expected in zip(fused, contiguous, strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.gpu
def test_broadcast_factors_keep_torch_path_on_cuda():
    q, k = torch.randn(2, 4, 3, 8, device="cuda").unbind(0)
    cos, sin = torch.randn(2, 1, 1, 8, device="cuda").unbind(0)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert torch.equal(q_rot, q * cos + rotate_half(q) * sin)
    assert torch.equal(k_rot, k * cos + rotate_half(k) * sin)

    heads, dim = INTERLEAVED_LAYOUT
    q, k = torch.randn(2, 3, heads, dim, device="cuda").unbind(0)
    freqs = torch.randn(1, 1, dim // 2, 2, device="cuda")

    fused = apply_interleaved_rotary_pos_emb(q, k, freqs)
    eager = apply_interleaved_rotary_pos_emb(q.cpu(), k.cpu(), freqs.cpu())

    for actual, expected in zip(fused, eager, strict=True):
        torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 512])
def test_fused_interleaved_rope_matches_torch_path(dtype, rows):
    heads, dim = INTERLEAVED_LAYOUT
    q = torch.randn(rows, heads, dim, device="cuda").to(dtype)
    k = torch.randn(rows, heads, dim, device="cuda").to(dtype)
    angles = torch.randint(0, 4096, (rows, 1, 1), device="cuda") * torch.rand(dim // 2).cuda()
    freqs = torch.stack((angles.cos(), angles.sin()), dim=-1)

    fused = apply_interleaved_rotary_pos_emb(q, k, freqs)
    eager = apply_interleaved_rotary_pos_emb(q.cpu(), k.cpu(), freqs.cpu())

    for actual, expected in zip(fused, eager, strict=True):
        assert actual.is_cuda and actual.dtype == dtype
        torch.testing.assert_close(actual.cpu(), expected, **_rounding_tolerance(dtype))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rope_graph_replay_is_bitwise_identical(dtype):
    rows, (query_heads, key_heads, dim) = 8, SPLIT_HALF_LAYOUTS[1]
    heads, interleaved_dim = INTERLEAVED_LAYOUT

    def make_inputs():
        cos, sin = _split_half_factors(rows, dim, dtype)
        angles = torch.rand(rows, 1, interleaved_dim // 2, device="cuda") * 100
        return (
            torch.randn(rows, query_heads, dim, device="cuda").to(dtype),
            torch.randn(rows, key_heads, dim, device="cuda").to(dtype),
            cos,
            sin,
            torch.randn(rows, heads, interleaved_dim, device="cuda").to(dtype),
            torch.randn(rows, heads, interleaved_dim, device="cuda").to(dtype),
            torch.stack((angles.cos(), angles.sin()), dim=-1),
        )

    def run(q, k, cos, sin, hq, hk, freqs):
        return (
            *apply_rotary_pos_emb(q, k, cos, sin),
            *apply_interleaved_rotary_pos_emb(hq, hk, freqs),
        )

    static = make_inputs()
    run(*static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run(*static)
    for _ in range(32):
        fresh = make_inputs()
        for buffer, value in zip(static, fresh, strict=True):
            buffer.copy_(value)
        graph.replay()
        for replayed, eager in zip(outputs, run(*fresh), strict=True):
            assert torch.equal(replayed, eager)
