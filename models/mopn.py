import torch
import torch.nn as nn

from models.blocks import RMSNorm, RotaryEmbedding, SwiGLUFeedForward


class MOPN_Layer(nn.Module):
    """
    Proyecta características en subespacios ortogonales usando subredes independientes.
    """

    def __init__(self, embed_dim: int, num_subspaces: int = 4, ff_multiplier: float = 2.0):
        super().__init__()
        if embed_dim % num_subspaces != 0:
            raise ValueError("embed_dim must be divisible by num_subspaces")

        self.norm = RMSNorm(embed_dim)
        self.rope = RotaryEmbedding(embed_dim)
        self.num_subspaces = num_subspaces
        self.subspace_dim = embed_dim // num_subspaces
        hidden_dim = max(self.subspace_dim, int(self.subspace_dim * ff_multiplier))
        self.sub_networks = nn.ModuleList(
            [
                SwiGLUFeedForward(
                    input_dim=self.subspace_dim,
                    hidden_dim=hidden_dim,
                    output_dim=self.subspace_dim,
                )
                for _ in range(num_subspaces)
            ]
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        hidden = self.rope.apply_to_hidden(hidden)
        chunks = torch.chunk(hidden, self.num_subspaces, dim=-1)
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            processed_chunks.append(self.sub_networks[i](chunk))

        out = torch.cat(processed_chunks, dim=-1)
        return self.output_proj(out) + x


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    embed_dim = 64

    x = torch.randn(batch_size, seq_len, embed_dim)
    model = MOPN_Layer(embed_dim=embed_dim, num_subspaces=4)
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape, f"Shape mismatch: in={x.shape}, out={out.shape}"
    print(f"OK - mopn: {x.shape} -> {out.shape}")
