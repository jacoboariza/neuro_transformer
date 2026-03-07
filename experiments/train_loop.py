from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.profiler import DeviceTimer


def _infer_embed_dim(model: nn.Module) -> int:
    explicit_embed_dim = getattr(model, "embed_dim", None)
    if isinstance(explicit_embed_dim, int) and explicit_embed_dim > 0:
        return explicit_embed_dim

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
    grad_accum_steps: int = 1,
) -> Dict[str, list]:
    """
    Entrena un modelo autoregresivo con datos (inputs, targets) para next-token prediction.

    - Optimizador: AdamW
    - Loss: CrossEntropyLoss
    - Si el modelo devuelve (batch, seq, embed), se proyecta a vocabulario con una
      capa lineal final dinámica (output_head) cuando no existe.
    - Soporta acumulación de gradientes (grad_accum_steps) para reducir uso de VRAM.
    - Si el modelo implementa sleep_cycle(), se ejecuta al final de cada epoch.
    """

    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    timer = DeviceTimer(device_t)

    history = {
        "loss": [],
        "avg_step_ms": [],
        "epoch_tokens": [],
        "epoch_compute_ms": [],
    }

    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Acumuladores en GPU
        total_loss_tensor = torch.tensor(0.0, device=device_t)
        
        batch_count = 0
        total_tokens = 0
        accumulation_counter = 0

        # Timer de epoch completo para medir throughput real (incluye carga de datos asíncrona oculta por pipelining)
        # Para medir solo cómputo GPU estricto, idealmente usaríamos eventos, pero para "train time" el usuario espera wall-clock o GPU-active time.
        # Usaremos el DeviceTimer para medir el bloque entero de ejecución del epoch.
        epoch_start_marker = timer.start()
        
        print(f"Epoch {epoch + 1}/{epochs} iniciando...")

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device_t, non_blocking=True)
            targets = targets.to(device_t, non_blocking=True)

            if needs_embedding:
                hidden = model(model.token_embedding(inputs))
            else:
                hidden = model(inputs)

            logits = output_head(hidden)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            (loss / grad_accum_steps).backward()
            accumulation_counter += 1

            if accumulation_counter >= grad_accum_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_counter = 0

            # Acumulación asíncrona
            total_loss_tensor += loss
            
            # targets.numel() es rápido en CPU (shape)
            total_tokens += int(targets.numel())
            batch_count += 1
            
            if batch_idx % 10 == 0 or (total_batches and batch_idx == total_batches - 1):
                if total_batches:
                    percent = (batch_idx + 1) / total_batches * 100
                    print(f"\r  Batch {batch_idx + 1}/{total_batches} ({percent:.1f}%) - Loss: {loss.item():.4f}", end="", flush=True)
                else:
                    print(f"\r  Batch {batch_idx + 1} - Loss: {loss.item():.4f}", end="", flush=True)

        if accumulation_counter > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        print() # Nueva línea al terminar el epoch
        
        # Sincronización única al final del epoch
        epoch_ms = timer.stop(epoch_start_marker)
        
        epoch_loss = float(total_loss_tensor.item()) / max(batch_count, 1)
        avg_step_ms = epoch_ms / max(batch_count, 1)
        
        history["loss"].append(epoch_loss)
        history["avg_step_ms"].append(avg_step_ms)
        history["epoch_tokens"].append(total_tokens)
        history["epoch_compute_ms"].append(epoch_ms)
        
        print(f"Epoch {epoch + 1} completado: Loss media={epoch_loss:.4f}, Tiempo={epoch_ms/1000:.2f}s")

        if hasattr(model, "sleep_cycle"):
            sleep_cycle_result = model.sleep_cycle()
            if sleep_cycle_result is not False:
                print("  Ciclo de sueño SCT ejecutado.")

    return history
