# SPDX-License-Identifier: Apache-2.0
"""Shared rotary position embedding (RoPE) layers across models."""

import torch
from torch import nn


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input tensor."""
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to Query and Key states.

    CUDA `[tokens, heads, dim]` inputs with per-token `[tokens, 1, dim]` factors use
    one fused Triton launch that rounds each output once; other inputs use Torch.
    """
    if (
        _fusable(q, k, cos, sin)
        and cos.shape == (q.shape[0], 1, q.shape[2])
        and cos.dtype == sin.dtype == q.dtype
    ):
        from vllm_rlt.kernels.rotary import split_half_rope

        return split_half_rope(q, k, cos, sin)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def apply_interleaved_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent (even, odd) pairs by `freqs[..., 0]` (cos) and `freqs[..., 1]` (sin).

    `q` and `k` are `[tokens, heads, dim]`; `freqs` is `[tokens, 1, dim / 2, 2]`.
    The rotation is computed in FP32 and returned in the input dtype.
    """
    rows, heads, dim = q.shape
    if (
        _fusable(q, k, freqs, freqs)
        and k.shape == q.shape
        and freqs.shape == (rows, 1, dim // 2, 2)
    ):
        from vllm_rlt.kernels.rotary import interleaved_rope

        return interleaved_rope(q, k, freqs)
    pair = torch.stack((q, k), dim=0).float().reshape(2, rows, heads, -1, 2)
    cos, sin = freqs[..., 0].unsqueeze(0), freqs[..., 1].unsqueeze(0)
    real = pair[..., 0] * cos - pair[..., 1] * sin
    imag = pair[..., 1] * cos + pair[..., 0] * sin
    q, k = torch.stack((real, imag), dim=-1).flatten(-2).to(q.dtype).unbind(0)
    return q, k


def _fusable(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and q.ndim == k.ndim == 3
        and q.shape[0] == k.shape[0]
        and q.shape[2] == k.shape[2]
        and q.dtype == k.dtype
        and cos.shape == sin.shape
        and cos.device == q.device
        and not (q.requires_grad or k.requires_grad)
    )


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with precision-preserving buffer management.

    Accepts `head_dim` and `rope_theta` directly without depending on a specific
    model configuration dataclass.
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.register_buffer("inv_freq", self.frequencies(), persistent=False)

    def frequencies(self, device: torch.device | str | None = None) -> torch.Tensor:
        target = device
        if device is not None and torch.device(device).type == "npu":
            device = "cpu"  # HF creates NPU rotary constants on the host.
        frequencies = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )

        return (
            frequencies.to(target)
            if device == "cpu" and target is not None and torch.device(target).type == "npu"
            else frequencies
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Module.to(dtype) otherwise rounds these constants before forward's
        # float32 cast, changing long-context rotary angles irreversibly.
        self.inv_freq = self.frequencies(device=self.inv_freq.device)
        return self

    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Force FP32 calculation to maintain accuracy regardless of model dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1)
        return (
            angles.cos().to(hidden.dtype).unsqueeze(1),
            angles.sin().to(hidden.dtype).unsqueeze(1),
        )
