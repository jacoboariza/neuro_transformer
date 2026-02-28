import torch
import torch.nn as nn

from models.blocks import RMSNorm, RotaryEmbedding, SwiGLUFeedForward


class SCT_Layer(nn.Module):
    """
    Simula vigilia (base + memoria corto plazo) y consolidación en ciclo de sueño.
    """

    def __init__(self, embed_dim: int, ff_multiplier: float = 2.0):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.rope = RotaryEmbedding(embed_dim)

        self.base_network = nn.Linear(embed_dim, embed_dim, bias=False)
        self.short_term_memory = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.short_term_memory.weight)

        hidden_dim = max(embed_dim, int(embed_dim * ff_multiplier))
        self.transition = SwiGLUFeedForward(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        hidden = self.rope.apply_to_hidden(hidden)
        hidden = self.base_network(hidden) + self.short_term_memory(hidden)
        return x + self.transition(hidden)

    def sleep_cycle(self, pruning_threshold: float = 0.01) -> None:
        with torch.no_grad():
            self.base_network.weight += self.short_term_memory.weight

            mask = torch.abs(self.base_network.weight) > pruning_threshold
            self.base_network.weight *= mask.float()

            self.short_term_memory.weight.zero_()


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    embed_dim = 64

    x = torch.randn(batch_size, seq_len, embed_dim)
    model = SCT_Layer(embed_dim=embed_dim)
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape, f"Shape mismatch: in={x.shape}, out={out.shape}"
    model.sleep_cycle()
    print(f"OK - sct: {x.shape} -> {out.shape}")
