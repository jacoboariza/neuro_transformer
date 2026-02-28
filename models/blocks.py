import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings usable in attention and dense token processing."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even dimension.")

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        needs_rebuild = (
            seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        )
        if not needs_rebuild:
            return

        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        angles = torch.cat((freqs, freqs), dim=-1)

        cos = angles.cos().to(dtype=dtype)
        sin = angles.sin().to(dtype=dtype)

        self._cos_cached = cos[None, None, :, :]
        self._sin_cached = sin[None, None, :, :]
        self._seq_len_cached = seq_len

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cache(seq_len=seq_len, device=device, dtype=dtype)
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.size(-1) != self.dim or k.size(-1) != self.dim:
            raise ValueError(f"RoPE dim mismatch. Expected {self.dim}, got q={q.size(-1)} k={k.size(-1)}")

        seq_len = q.size(-2)
        cos, sin = self._cos_sin(seq_len=seq_len, device=q.device, dtype=q.dtype)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

    def apply_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.dim:
            raise ValueError(f"RoPE dim mismatch. Expected {self.dim}, got hidden={x.size(-1)}")

        hidden = x.unsqueeze(1)
        seq_len = hidden.size(-2)
        cos, sin = self._cos_sin(seq_len=seq_len, device=hidden.device, dtype=hidden.dtype)
        rotated = (hidden * cos) + (rotate_half(hidden) * sin)
        return rotated.squeeze(1)


class SwiGLUFeedForward(nn.Module):
    """Feed-forward block with SwiGLU activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        gated = self.dropout(gated)
        return self.down_proj(gated)


class MultiHeadSelfAttentionRoPE(nn.Module):
    """Causal self-attention with RoPE and projection head."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even to apply RoPE")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rotary = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary.apply_rotary(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(context)
        return self.resid_dropout(out)
