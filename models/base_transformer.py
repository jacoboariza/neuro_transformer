import torch
import torch.nn as nn

from models.blocks import MultiHeadSelfAttentionRoPE, RMSNorm, SwiGLUFeedForward


class StandardTransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attention = MultiHeadSelfAttentionRoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = SwiGLUFeedForward(
            input_dim=embed_dim,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), causal=True)
        x = x + self.ffn(self.ffn_norm(x))
        return x


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    embed_dim = 64

    x = torch.randn(batch_size, seq_len, embed_dim)
    model = StandardTransformerLayer(embed_dim=embed_dim, num_heads=8, ff_dim=256)
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape, f"Shape mismatch: in={x.shape}, out={out.shape}"
    print(f"OK - base_transformer: {x.shape} -> {out.shape}")
