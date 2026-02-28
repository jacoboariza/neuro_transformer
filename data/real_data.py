from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from datasets import Dataset, load_dataset
except ImportError:  # pragma: no cover
    Dataset = None
    load_dataset = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


def _load_dataset_split(dataset_name: str, dataset_config: Optional[str], split: str):
    if load_dataset is None:
        raise ImportError("datasets no está instalado. Agrega 'datasets' al entorno.")

    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    return load_dataset(dataset_name, split=split)


def load_real_dataset_subset(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: Optional[str] = "sample-10BT",
    split: str = "train",
    num_samples: int = 50_000,
    seed: int = 42,
    fallback_dataset_name: str = "wikimedia/wikipedia",
    fallback_dataset_config: Optional[str] = "20231101.es",
):
    """
    Descarga un subconjunto reproducible de un dataset real de texto.

    Se intenta primero el dataset principal y, si falla, se usa una alternativa
    de Wikipedia en español para mantener el pipeline operativo.
    """

    candidates = [(dataset_name, dataset_config)]
    if (fallback_dataset_name, fallback_dataset_config) != (dataset_name, dataset_config):
        candidates.append((fallback_dataset_name, fallback_dataset_config))

    last_error = None
    for candidate_name, candidate_config in candidates:
        try:
            dataset = _load_dataset_split(
                dataset_name=candidate_name,
                dataset_config=candidate_config,
                split=split,
            )
            print(f"Dataset cargado: {candidate_name} ({candidate_config})")
            break
        except Exception as exc:  # pragma: no cover - depende del entorno
            last_error = exc
            print(f"No se pudo cargar {candidate_name} ({candidate_config}): {exc}")
    else:
        raise RuntimeError(f"No se pudo cargar ningún dataset real: {last_error}")

    if num_samples is not None and num_samples > 0 and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    return dataset


def load_text_tokenizer(tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"):
    if AutoTokenizer is None:
        raise ImportError("transformers no está instalado. Agrega 'transformers' al entorno.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def infer_text_column(dataset, explicit_text_column: Optional[str] = None) -> str:
    if explicit_text_column:
        if explicit_text_column not in dataset.column_names:
            raise ValueError(
                f"text_column='{explicit_text_column}' no existe en columnas: {dataset.column_names}"
            )
        return explicit_text_column

    preferred = ("text", "content", "document", "article", "body")
    for column in preferred:
        if column in dataset.column_names:
            return column

    sample = dataset[0]
    for column, value in sample.items():
        if isinstance(value, str):
            return column

    raise ValueError(f"No se encontró una columna de texto. Columnas: {dataset.column_names}")


def build_token_windows(
    dataset,
    tokenizer,
    seq_len: int,
    text_column: str,
    num_proc: Optional[int] = None,
    add_eos_token: bool = True,
) -> torch.Tensor:
    """
    Tokeniza un dataset real y agrupa en ventanas de longitud (seq_len + 1).
    """
    if seq_len <= 1:
        raise ValueError("seq_len must be > 1")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id

    block_size = seq_len + 1

    def tokenize_batch(batch):
        texts = []
        for text in batch[text_column]:
            if isinstance(text, str):
                cleaned = text.strip()
                if cleaned:
                    texts.append(cleaned)

        if not texts:
            return {"input_ids": []}

        tokenized = tokenizer(texts, add_special_tokens=False, truncation=False)
        token_ids = tokenized["input_ids"]

        if add_eos_token and eos_id is not None:
            token_ids = [ids + [eos_id] for ids in token_ids]

        return {"input_ids": token_ids}

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizando dataset real",
    )

    def group_tokens(batch):
        flat_tokens = []
        for row in batch["input_ids"]:
            flat_tokens.extend(row)

        total_len = (len(flat_tokens) // block_size) * block_size
        if total_len == 0:
            return {"input_ids": []}

        flat_tokens = flat_tokens[:total_len]
        grouped = [
            flat_tokens[start : start + block_size]
            for start in range(0, total_len, block_size)
        ]
        return {"input_ids": grouped}

    grouped_dataset = tokenized_dataset.map(
        group_tokens,
        batched=True,
        desc="Agrupando secuencias",
    )

    if len(grouped_dataset) == 0:
        raise ValueError("No se pudieron construir ventanas de tokens. Incrementa num_samples o reduce seq_len.")

    windows = torch.tensor(grouped_dataset["input_ids"], dtype=torch.long)
    return windows


def create_real_dataloaders(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: Optional[str] = "sample-10BT",
    split: str = "train",
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M",
    text_column: Optional[str] = None,
    num_samples: int = 50_000,
    seq_len: int = 512,
    batch_size: int = 8,
    val_split: float = 0.05,
    seed: int = 42,
    num_proc: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, object]:
    """
    Construye DataLoaders train/val para next-token prediction con datos reales.
    """
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in (0.0, 1.0)")

    tokenizer = load_text_tokenizer(tokenizer_name=tokenizer_name)
    dataset = load_real_dataset_subset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        num_samples=num_samples,
        seed=seed,
    )

    resolved_text_column = infer_text_column(dataset, explicit_text_column=text_column)
    print(f"Columna de texto usada: {resolved_text_column}")

    windows = build_token_windows(
        dataset=dataset,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column=resolved_text_column,
        num_proc=num_proc,
    )

    total_windows = windows.size(0)
    if total_windows < 2:
        raise ValueError("Se requieren al menos 2 ventanas para separar train/val.")

    val_size = max(1, int(total_windows * val_split))
    val_size = min(val_size, total_windows - 1)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_windows, generator=generator)
    windows = windows[permutation]

    train_windows = windows[:-val_size]
    val_windows = windows[-val_size:]

    train_dataset = TensorDataset(train_windows[:, :-1], train_windows[:, 1:])
    val_dataset = TensorDataset(val_windows[:, :-1], val_windows[:, 1:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(
        f"Ventanas totales={total_windows} train={len(train_dataset)} val={len(val_dataset)} "
        f"seq_len={seq_len} vocab={len(tokenizer)}"
    )

    return train_loader, val_loader, tokenizer
