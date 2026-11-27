"""Fused query/key rotary embedding kernels.

One program rotates one (token, head) row. Query heads come first, then key
heads, so a single launch replaces the separate elementwise operations for both
tensors. Inputs are read in their own dtype, the rotation is computed in FP32
and each output element is rounded to the activation dtype once.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_half_rope(
    Q,
    K,
    Cos,
    Sin,
    QOut,
    KOut,
    q_row,
    q_head,
    q_dim,
    k_row,
    k_head,
    k_dim,
    cos_row,
    cos_dim,
    sin_row,
    sin_dim,
    QUERY_HEADS: tl.constexpr,
    KEY_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    first = tl.arange(0, BLOCK)
    valid = first < HALF
    second = first + HALF
    cos_lo = tl.load(Cos + row * cos_row + first * cos_dim, mask=valid).to(tl.float32)
    cos_hi = tl.load(Cos + row * cos_row + second * cos_dim, mask=valid).to(tl.float32)
    sin_lo = tl.load(Sin + row * sin_row + first * sin_dim, mask=valid).to(tl.float32)
    sin_hi = tl.load(Sin + row * sin_row + second * sin_dim, mask=valid).to(tl.float32)
    if head < QUERY_HEADS:
        source = Q + row * q_row + head * q_head
        target = QOut + (row * QUERY_HEADS + head) * 2 * HALF
        stride = q_dim
    else:
        source = K + row * k_row + (head - QUERY_HEADS) * k_head
        target = KOut + (row * KEY_HEADS + head - QUERY_HEADS) * 2 * HALF
        stride = k_dim
    lo = tl.load(source + first * stride, mask=valid).to(tl.float32)
    hi = tl.load(source + second * stride, mask=valid).to(tl.float32)
    out_lo = lo * cos_lo - hi * sin_lo
    out_hi = hi * cos_hi + lo * sin_hi
    tl.store(target + first, out_lo.to(target.dtype.element_ty), mask=valid)
    tl.store(target + second, out_hi.to(target.dtype.element_ty), mask=valid)


@triton.jit
def _interleaved_rope(
    Q,
    K,
    Freqs,
    QOut,
    KOut,
    q_row,
    q_head,
    q_dim,
    k_row,
    k_head,
    k_dim,
    f_row,
    f_pair,
    f_part,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    pair = tl.arange(0, BLOCK)
    valid = pair < HALF
    cos = tl.load(Freqs + row * f_row + pair * f_pair, mask=valid).to(tl.float32)
    sin = tl.load(Freqs + row * f_row + pair * f_pair + f_part, mask=valid).to(tl.float32)
    if head < HEADS:
        source = Q + row * q_row + head * q_head
        target = QOut + (row * HEADS + head) * 2 * HALF
        stride = q_dim
    else:
        source = K + row * k_row + (head - HEADS) * k_head
        target = KOut + (row * HEADS + head - HEADS) * 2 * HALF
        stride = k_dim
    real = tl.load(source + 2 * pair * stride, mask=valid).to(tl.float32)
    imag = tl.load(source + (2 * pair + 1) * stride, mask=valid).to(tl.float32)
    tl.store(target + 2 * pair, (real * cos - imag * sin).to(target.dtype.element_ty), mask=valid)
    tl.store(
        target + 2 * pair + 1, (imag * cos + real * sin).to(target.dtype.element_ty), mask=valid
    )


def split_half_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate `[tokens, heads, dim]` queries and keys by `[tokens, 1, dim]` factors."""
    rows, query_heads, dim = q.shape
    key_heads = k.shape[1]
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if rows:
        _split_half_rope[(rows, query_heads + key_heads)](
            q,
            k,
            cos,
            sin,
            q_out,
            k_out,
            *q.stride(),
            *k.stride(),
            cos.stride(0),
            cos.stride(2),
            sin.stride(0),
            sin.stride(2),
            QUERY_HEADS=query_heads,
            KEY_HEADS=key_heads,
            HALF=dim // 2,
            BLOCK=triton.next_power_of_2(dim // 2),
        )
    return q_out, k_out


def interleaved_rope(
    q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent pairs of `[tokens, heads, dim]` by `[tokens, 1, dim/2, 2]` cos/sin."""
    rows, heads, dim = q.shape
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if rows:
        _interleaved_rope[(rows, 2 * heads)](
            q,
            k,
            freqs,
            q_out,
            k_out,
            *q.stride(),
            *k.stride(),
            freqs.stride(0),
            freqs.stride(2),
            freqs.stride(3),
            HEADS=heads,
            HALF=dim // 2,
            BLOCK=triton.next_power_of_2(dim // 2),
        )
    return q_out, k_out
