from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_transformer import StandardTransformerLayer
from models.blocks import RMSNorm
from models.dca import DCA_Layer
from models.gma_moe import GMA_MoE_Layer
from models.mopn import MOPN_Layer
from models.sct import SCT_Layer


LAYER_REGISTRY: Dict[str, Type[nn.Module]] = {
    "transformer": StandardTransformerLayer,
    "dca": DCA_Layer,
    "mopn": MOPN_Layer,
    "sct": SCT_Layer,
    "gma_moe": GMA_MoE_Layer,
}


def _default_layer_kwargs(layer_type: str, embed_dim: int) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}

    if layer_type == "transformer":
        defaults["num_heads"] = 8 if embed_dim % 8 == 0 else 4
        defaults["ff_dim"] = embed_dim * 4

    return defaults


class NeuroModel(nn.Module):
    """
    Wrapper multicapa para apilar N bloques bio-inspirados y proyectar a vocabulario.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 12,
        layer_type: str = "transformer",
        layer_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
    ):
        super().__init__()

        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        layer_key = layer_type.lower()
        if layer_key not in LAYER_REGISTRY:
            supported = ", ".join(sorted(LAYER_REGISTRY.keys()))
            raise ValueError(f"layer_type '{layer_type}' no soportado. Opciones: {supported}")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.layer_type = layer_key

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        resolved_layer_kwargs = _default_layer_kwargs(layer_key, embed_dim)
        if layer_kwargs:
            resolved_layer_kwargs.update(layer_kwargs)

        layer_cls = LAYER_REGISTRY[layer_key]
        self.layers = nn.ModuleList(
            [layer_cls(embed_dim=embed_dim, **resolved_layer_kwargs) for _ in range(num_layers)]
        )

        self.final_norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if input_ids.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            hidden = self.token_embedding(input_ids)
        else:
            hidden = input_ids

        hidden = self.embedding_dropout(hidden)
        for layer in self.layers:
            hidden = layer(hidden)

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
