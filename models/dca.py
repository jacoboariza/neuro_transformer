import torch
import torch.nn as nn
import math

from models.blocks import RMSNorm, RotaryEmbedding, SwiGLUFeedForward


class FixedSparseLinear(nn.Module):
    """
    Capa lineal con conectividad fija sparse usando torch.sparse.

    Implementa y = x W^T + b sin materializar una matriz densa en el forward.
    """

    def __init__(self, in_features: int, out_features: int, sparsity: float, bias: bool = True):
        super().__init__()
        if not (0.0 <= sparsity < 1.0):
            raise ValueError("sparsity must be in [0.0, 1.0)")

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        total_connections = in_features * out_features
        nnz = max(1, int(total_connections * (1.0 - sparsity)))

        with torch.no_grad():
            flat_indices = torch.randperm(total_connections)[:nnz]
            flat_indices, _ = flat_indices.sort()
            row_indices = flat_indices // in_features
            col_indices = flat_indices % in_features
            sparse_indices = torch.stack([row_indices, col_indices], dim=0)

        self.register_buffer("sparse_indices", sparse_indices)
        self.sparse_values = nn.Parameter(torch.empty(nnz))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._cached_weight = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(1, self.in_features))
        nn.init.uniform_(self.sparse_values, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        self._cached_weight = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_weight = None

    def sparse_weight(self) -> torch.Tensor:
        if not self.training and self._cached_weight is not None:
            # Verificar si el device coincide por si hubo movimiento
            if self._cached_weight.device == self.sparse_values.device:
                return self._cached_weight
            self._cached_weight = None

        weight = torch.sparse_coo_tensor(
            self.sparse_indices,
            self.sparse_values,
            size=(self.out_features, self.in_features),
            device=self.sparse_values.device,
            dtype=self.sparse_values.dtype,
        ).coalesce()
        
        if not self.training:
            self._cached_weight = weight
            
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x2d = x.reshape(-1, self.in_features)

        sparse_weight = self.sparse_weight()
        projected = torch.sparse.mm(sparse_weight, x2d.to(dtype=sparse_weight.dtype).transpose(0, 1)).transpose(0, 1)

        if self.bias is not None:
            projected = projected + self.bias

        projected = projected.reshape(*original_shape[:-1], self.out_features)
        if projected.dtype != x.dtype:
            projected = projected.to(dtype=x.dtype)
        return projected

    @property
    def density(self) -> float:
        total = float(self.in_features * self.out_features)
        return float(self.sparse_values.numel()) / max(total, 1.0)


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

        self.sparse_linear = FixedSparseLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            sparsity=sparsity,
            bias=False,
        )
        hidden_dim = max(embed_dim, int(embed_dim * ff_multiplier))
        self.swiglu = SwiGLUFeedForward(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
        )
        self.linear2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sparsity = sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        hidden = self.rope.apply_to_hidden(hidden)

        out = self.sparse_linear(hidden)
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
