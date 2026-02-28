import torch
import torch.nn as nn

from models.blocks import RMSNorm, RotaryEmbedding, SwiGLUFeedForward


class GMA_MoE_Layer(nn.Module):
    """
    Red glial estima complejidad y activa dinámicamente expertos.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        ff_multiplier: float = 2.0,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")

        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))

        self.norm = RMSNorm(embed_dim)
        self.rope = RotaryEmbedding(embed_dim)
        self.glial_router = nn.Linear(embed_dim, num_experts, bias=False)

        hidden_dim = max(embed_dim, int(embed_dim * ff_multiplier))
        self.experts = nn.ModuleList(
            [
                SwiGLUFeedForward(
                    input_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_dim=embed_dim,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        hidden = self.rope.apply_to_hidden(hidden)

        router_logits = self.glial_router(hidden)
        router_probs = torch.softmax(router_logits, dim=-1)

        if self.top_k < self.num_experts:
            top_values, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
            sparse_probs = torch.zeros_like(router_probs).scatter_(-1, top_indices, top_values)
            router_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        mixed = torch.sum(router_probs.unsqueeze(-1) * expert_outputs, dim=-2)

        return mixed + x


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    embed_dim = 64

    x = torch.randn(batch_size, seq_len, embed_dim)
    model = GMA_MoE_Layer(embed_dim=embed_dim, num_experts=4)
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape, f"Shape mismatch: in={x.shape}, out={out.shape}"
    print(f"OK - gma_moe: {x.shape} -> {out.shape}")
