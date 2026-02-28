import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import RMSNorm, RotaryEmbedding, SwiGLUFeedForward


class DCA_Layer(nn.Module):
    """
    Imita el conectoma con enrutamiento escaso, evitando atención densa NxN.
    """

    def __init__(
        self,
        embed_dim: int,
        sparsity: float = 0.8,
        ff_multiplier: float = 2.0,
    ):
        super().__init__()
        if not (0.0 <= sparsity < 1.0):
            raise ValueError("sparsity must be in [0.0, 1.0)")

        self.norm = RMSNorm(embed_dim)
        self.rope = RotaryEmbedding(embed_dim)

        self.linear1 = nn.Linear(embed_dim, embed_dim, bias=False)
        hidden_dim = max(embed_dim, int(embed_dim * ff_multiplier))
        self.swiglu = SwiGLUFeedForward(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
        )
        self.linear2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sparsity = sparsity

        with torch.no_grad():
            mask = (torch.rand(embed_dim, embed_dim) > sparsity).float()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        hidden = self.rope.apply_to_hidden(hidden)

        sparse_weight1 = self.linear1.weight * self.mask
        out = F.linear(hidden, sparse_weight1, self.linear1.bias)
        out = self.swiglu(out)
        out = self.linear2(out)
        return out + x


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    embed_dim = 64

    x = torch.randn(batch_size, seq_len, embed_dim)
    model = DCA_Layer(embed_dim=embed_dim, sparsity=0.85)
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape, f"Shape mismatch: in={x.shape}, out={out.shape}"
    print(f"OK - dca: {x.shape} -> {out.shape}")
