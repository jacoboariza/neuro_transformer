from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _infer_embed_dim(model: nn.Module) -> int:
    if hasattr(model, "attention") and hasattr(model.attention, "embed_dim"):
        return int(model.attention.embed_dim)

    if hasattr(model, "output_proj") and isinstance(model.output_proj, nn.Linear):
        return int(model.output_proj.in_features)

    if hasattr(model, "subspace_dim") and hasattr(model, "num_subspaces"):
        return int(model.subspace_dim) * int(model.num_subspaces)

    if hasattr(model, "linear1") and isinstance(model.linear1, nn.Linear):
        return int(model.linear1.in_features)

    if hasattr(model, "base_network") and isinstance(model.base_network, nn.Linear):
        return int(model.base_network.in_features)

    if hasattr(model, "experts") and len(model.experts) > 0 and isinstance(model.experts[0], nn.Linear):
        return int(model.experts[0].in_features)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)

    raise ValueError("No se pudo inferir embed_dim desde el modelo.")


def _infer_vocab_size(dataloader: DataLoader) -> int:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        dataset_vocab = getattr(dataset, "vocab_size", None)
        if isinstance(dataset_vocab, int) and dataset_vocab > 1:
            return dataset_vocab

        dataset_tokens = getattr(dataset, "_tokens", None)
        if isinstance(dataset_tokens, torch.Tensor):
            return int(dataset_tokens.max().item()) + 1

    max_token = 0
    for _, targets in dataloader:
        max_token = max(max_token, int(targets.max().item()))
    return max_token + 1


def _ensure_output_head(model: nn.Module, hidden_dim: int, vocab_size: int, device: torch.device) -> nn.Linear:
    candidate_names = (
        "lm_head",
        "output_head",
        "output_projection",
        "vocab_projection",
        "classifier",
    )

    for name in candidate_names:
        module = getattr(model, name, None)
        if isinstance(module, nn.Linear) and module.in_features == hidden_dim and module.out_features == vocab_size:
            return module.to(device)

    head = nn.Linear(hidden_dim, vocab_size).to(device)
    model.output_head = head
    return head


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> Dict[str, list]:
    """
    Entrena un modelo autoregresivo con datos (inputs, targets) para next-token prediction.

    - Optimizador: Adam
    - Loss: CrossEntropyLoss
    - Si el modelo devuelve (batch, seq, embed), se proyecta a vocabulario con una
      capa lineal final dinámica (output_head) cuando no existe.
    - Para SCT_Layer se ejecuta sleep_cycle() al final de cada epoch.
    """

    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if lr <= 0:
        raise ValueError("lr must be > 0")

    device_t = torch.device(device)
    model = model.to(device_t)
    vocab_size = _infer_vocab_size(dataloader)
    embed_dim = _infer_embed_dim(model)

    sample_inputs, _ = next(iter(dataloader))
    sample_inputs = sample_inputs.to(device_t)

    needs_embedding = sample_inputs.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )

    if needs_embedding:
        token_embedding = getattr(model, "token_embedding", None)
        if (
            not isinstance(token_embedding, nn.Embedding)
            or token_embedding.embedding_dim != embed_dim
            or token_embedding.num_embeddings < vocab_size
        ):
            model.token_embedding = nn.Embedding(vocab_size, embed_dim).to(device_t)

    model.train()
    criterion = nn.CrossEntropyLoss()

    # Inferimos hidden_dim real en una pasada para construir/validar output head.
    with torch.no_grad():
        if needs_embedding:
            sample_hidden = model(model.token_embedding(sample_inputs))
        else:
            sample_hidden = model(sample_inputs)

    if sample_hidden.dim() != 3:
        raise ValueError("Se esperaba salida del modelo con shape (batch, seq, hidden).")

    hidden_dim = int(sample_hidden.size(-1))
    output_head = _ensure_output_head(model, hidden_dim, vocab_size, device_t)
    output_head.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": []}

    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device_t)
            targets = targets.to(device_t)

            optimizer.zero_grad()

            if needs_embedding:
                hidden = model(model.token_embedding(inputs))
            else:
                hidden = model(inputs)

            logits = output_head(hidden)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        epoch_loss = total_loss / max(batch_count, 1)
        history["loss"].append(epoch_loss)

        if model.__class__.__name__ == "SCT_Layer" and hasattr(model, "sleep_cycle"):
            model.sleep_cycle()

    return history
