from pathlib import Path
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - entorno sin transformers
    AutoModelForCausalLM = None
    AutoTokenizer = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.real_data import create_real_dataloaders
from experiments.train_loop import train_model
from models.base_transformer import StandardTransformerLayer
from models.dca import DCA_Layer
from models.gma_moe import GMA_MoE_Layer
from models.mopn import MOPN_Layer
from models.sct import SCT_Layer
from utils.compliance import CANONICAL_DATASET, CANONICAL_TOKENIZER, build_compliance_report
from utils.profiler import DeviceTimer, estimate_flops_torch_profiler


REAL_CASE_TEXTS = [
    "In distributed systems, consistency and availability often conflict under network partitions.",
    "La plasticidad sinaptica permite adaptar representaciones internas ante estimulos cambiantes.",
    "The benchmark should report both quality metrics and computational efficiency under equal settings.",
    "Durante la consolidacion nocturna, las conexiones debiles se reducen y los patrones robustos se preservan.",
    "Sparse routing can reduce redundant computation while preserving useful pathways in sequence models.",
    "Un modelo robusto debe generalizar en datos no vistos y mantener estabilidad frente a ruido estructurado.",
    "Researchers compare latency, memory footprint, and predictive performance before selecting a final architecture.",
    "El uso de casos reales complementa los datos sinteticos y mejora la validez externa de los resultados.",
]


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def safe_div(value: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return value / denominator


def _normalize_score_series(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    normalized = pd.Series(float("nan"), index=values.index, dtype=float)
    valid = values.dropna()

    if valid.empty:
        return normalized

    min_value = float(valid.min())
    max_value = float(valid.max())
    span = max_value - min_value

    if span < 1e-12:
        normalized.loc[valid.index] = 1.0
        return normalized

    if higher_is_better:
        normalized.loc[valid.index] = (valid - min_value) / span
    else:
        normalized.loc[valid.index] = (max_value - valid) / span

    return normalized


def _weighted_score(df: pd.DataFrame, metric_specs: List[Tuple[str, bool, float]]) -> pd.Series:
    weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
    total_weight = pd.Series(0.0, index=df.index, dtype=float)

    for metric_name, higher_is_better, weight in metric_specs:
        if metric_name not in df.columns:
            continue

        normalized = _normalize_score_series(df[metric_name], higher_is_better)
        valid_mask = normalized.notna()

        weighted_sum.loc[valid_mask] += normalized.loc[valid_mask] * weight
        total_weight.loc[valid_mask] += weight

    score = pd.Series(float("nan"), index=df.index, dtype=float)
    valid_rows = total_weight > 0
    score.loc[valid_rows] = weighted_sum.loc[valid_rows] / total_weight.loc[valid_rows]
    return score


def add_composite_ranking(results_df: pd.DataFrame) -> pd.DataFrame:
    ranked = results_df.copy()

    quality_metric_specs: List[Tuple[str, bool, float]] = [
        ("RealCaseAccuracy", True, 0.45),
        ("RealCaseLoss", False, 0.35),
        ("FinalLoss", False, 0.20),
    ]
    efficiency_metric_specs: List[Tuple[str, bool, float]] = [
        ("TrainTokensPerSecond", True, 0.60),
        ("SecondsPerMParam", False, 0.25),
        ("TrainTimeSeconds", False, 0.15),
    ]

    quality_score = _weighted_score(ranked, quality_metric_specs)
    efficiency_score = _weighted_score(ranked, efficiency_metric_specs)

    ranked["QualityScore"] = quality_score * 100.0
    ranked["EfficiencyScore"] = efficiency_score * 100.0

    composite_df = pd.DataFrame(
        {
            "QualityScore": ranked["QualityScore"],
            "EfficiencyScore": ranked["EfficiencyScore"],
        }
    )
    composite_score = _weighted_score(
        composite_df,
        [
            ("QualityScore", True, 0.65),
            ("EfficiencyScore", True, 0.35),
        ],
    )
    ranked["CompositeScore"] = composite_score

    if "EligibleForRanking" in ranked.columns:
        ineligible = ~ranked["EligibleForRanking"].fillna(False).astype(bool)
        ranked.loc[ineligible, "CompositeScore"] = float("nan")

    rank_series = ranked["CompositeScore"].rank(method="dense", ascending=False)
    rank_series[ranked["CompositeScore"].isna()] = pd.NA
    ranked["CompositeRank"] = rank_series.astype("Int64")

    return ranked


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def count_tokens_per_epoch(dataloader: DataLoader) -> int:
    dataset = getattr(dataloader, "dataset", None)
    if isinstance(dataset, TensorDataset):
        return int(dataset.tensors[0].numel())

    total_tokens = 0
    for inputs, _ in dataloader:
        total_tokens += int(inputs.numel())
    return total_tokens


def build_real_case_dataloader(
    texts: List[str],
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    repeats: int = 12,
) -> DataLoader:
    expanded_texts = texts * max(1, repeats)
    tokenized_texts = [_simple_tokenize(text) for text in expanded_texts]

    token_counts = Counter(token for tokens in tokenized_texts for token in tokens)
    special_tokens = ["<pad>", "<unk>", "<eos>"]
    max_vocab = max(vocab_size, len(special_tokens) + 1)

    token_to_id = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _ in token_counts.most_common(max_vocab - len(special_tokens)):
        if token in token_to_id:
            continue
        token_to_id[token] = len(token_to_id)
        if len(token_to_id) >= max_vocab:
            break

    eos_id = token_to_id["<eos>"]
    unk_id = token_to_id["<unk>"]

    token_stream: List[int] = []
    for tokens in tokenized_texts:
        if not tokens:
            continue
        token_stream.extend(token_to_id.get(token, unk_id) for token in tokens)
        token_stream.append(eos_id)

    window_size = seq_len + 1
    if not token_stream:
        token_stream = [eos_id] * window_size

    while len(token_stream) < window_size:
        missing = window_size - len(token_stream)
        token_stream.extend(token_stream[:missing])

    stride = max(1, seq_len // 2)
    windows = []
    for start in range(0, len(token_stream) - window_size + 1, stride):
        windows.append(token_stream[start : start + window_size])

    if not windows:
        windows = [token_stream[:window_size]]

    tokens_tensor = torch.tensor(windows, dtype=torch.long)
    dataset = TensorDataset(tokens_tensor[:, :-1], tokens_tensor[:, 1:])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_reasoner_real_case_dataloader(
    tokenizer,
    texts: List[str],
    seq_len: int,
    batch_size: int,
    repeats: int = 12,
) -> DataLoader:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    token_stream: List[int] = []
    for text in texts * max(1, repeats):
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if not token_ids:
            continue
        token_stream.extend(token_ids)
        token_stream.append(eos_id)

    window_size = seq_len + 1
    if not token_stream:
        token_stream = [eos_id] * window_size

    while len(token_stream) < window_size:
        missing = window_size - len(token_stream)
        token_stream.extend(token_stream[:missing])

    stride = max(1, seq_len // 2)
    windows = []
    for start in range(0, len(token_stream) - window_size + 1, stride):
        windows.append(token_stream[start : start + window_size])

    if not windows:
        windows = [token_stream[:window_size]]

    tokens_tensor = torch.tensor(windows, dtype=torch.long)
    dataset = TensorDataset(tokens_tensor[:, :-1], tokens_tensor[:, 1:])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_custom_model_on_real_cases(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    if not hasattr(model, "token_embedding") or not hasattr(model, "output_head"):
        return {
            "RealCaseLoss": float("nan"),
            "RealCaseAccuracy": float("nan"),
            "RealCaseEvalSeconds": float("nan"),
        }

    device_t = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    timer = DeviceTimer(device_t)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batch_count = 0
    total_eval_ms = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device_t)
            targets = targets.to(device_t)

            eval_start = timer.start()

            hidden = model(model.token_embedding(inputs))
            logits = model.output_head(hidden)

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            predictions = logits.argmax(dim=-1)

            total_loss += float(loss.item())
            total_correct += int((predictions == targets).sum().item())
            total_tokens += int(targets.numel())
            batch_count += 1
            total_eval_ms += timer.stop(eval_start)

    elapsed = total_eval_ms / 1000.0
    return {
        "RealCaseLoss": safe_div(total_loss, max(batch_count, 1)),
        "RealCaseAccuracy": safe_div(total_correct, max(total_tokens, 1)),
        "RealCaseEvalSeconds": elapsed,
    }


def evaluate_reasoner_on_real_cases(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    device_t = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    timer = DeviceTimer(device_t)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batch_count = 0
    total_eval_ms = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device_t)
            targets = targets.to(device_t)

            eval_start = timer.start()

            outputs = model(input_ids=inputs)
            logits = outputs.logits

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            predictions = logits.argmax(dim=-1)

            total_loss += float(loss.item())
            total_correct += int((predictions == targets).sum().item())
            total_tokens += int(targets.numel())
            batch_count += 1
            total_eval_ms += timer.stop(eval_start)

    elapsed = total_eval_ms / 1000.0
    return {
        "RealCaseLoss": safe_div(total_loss, max(batch_count, 1)),
        "RealCaseAccuracy": safe_div(total_correct, max(total_tokens, 1)),
        "RealCaseEvalSeconds": elapsed,
    }


def load_local_reasoner(
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    device: str = "cpu",
) -> Tuple[object, torch.nn.Module]:
    """
    Carga un reasoner local de Hugging Face para incluirlo en la comparativa.
    """
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("transformers no está instalado en el entorno actual.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.train()
    return tokenizer, model


def train_reasoner_model(
    model: torch.nn.Module,
    dataloader,
    epochs: int,
    lr: float,
    device: str,
) -> Dict[str, list]:
    """
    Entrenamiento simple para CausalLM de Hugging Face con targets shiftados.
    """
    device_t = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    timer = DeviceTimer(device_t)
    history = {
        "loss": [],
        "avg_step_ms": [],
        "epoch_tokens": [],
        "epoch_compute_ms": [],
    }

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        total_step_ms = 0.0
        total_tokens = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device_t)
            targets = targets.to(device_t)

            step_start = timer.start()

            optimizer.zero_grad()
            outputs = model(input_ids=inputs)
            logits = outputs.logits

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            step_ms = timer.stop(step_start)

            total_loss += float(loss.item())
            batch_count += 1
            total_step_ms += step_ms
            total_tokens += int(targets.numel())

        history["loss"].append(total_loss / max(batch_count, 1))
        history["avg_step_ms"].append(total_step_ms / max(batch_count, 1))
        history["epoch_tokens"].append(total_tokens)
        history["epoch_compute_ms"].append(total_step_ms)

    return history


def benchmark_local_reasoner(
    dataloader,
    validation_dataloader,
    epochs: int,
    lr: float,
    device: str,
    train_tokens_per_epoch: int,
    requested_dataset_name: str,
    requested_dataset_config: Optional[str],
    resolved_dataset_name: str,
    resolved_dataset_config: Optional[str],
    requested_tokenizer_name: str,
    resolved_tokenizer_name: str,
    used_fallback_dataset: bool,
    evaluate_real_cases: bool = True,
    model_name: str = "HuggingFaceTB/SmolLM-135M",
) -> Optional[dict]:
    device_t = torch.device(device)
    profiling_backend = "cuda_event" if device_t.type == "cuda" else "perf_counter"

    try:
        _tokenizer, reasoner = load_local_reasoner(model_name=model_name, device=device)
    except Exception as exc:
        print(f"[SmolLM] No se pudo cargar {model_name}: {exc}")
        return {
            "Model": "SmolLM-135M",
            "FinalLoss": float("nan"),
            "TrainableParams": float("nan"),
            "TrainTimeSeconds": float("nan"),
            "TrainTokens": float("nan"),
            "TrainTokensPerSecond": float("nan"),
            "SecondsPerMParam": float("nan"),
            "TrainFLOPs": float("nan"),
            "RealCaseLoss": float("nan"),
            "RealCaseAccuracy": float("nan"),
            "RealCaseEvalSeconds": float("nan"),
            "R1_RealData": False,
            "R2_SparseDCA": False,
            "R3_TokenMaskingPMT": False,
            "R4_VLMDetach": False,
            "R5_PreciseProfiling": False,
            "EligibleForRanking": False,
            "RequestedDatasetName": requested_dataset_name,
            "RequestedDatasetConfig": requested_dataset_config,
            "ResolvedDatasetName": resolved_dataset_name,
            "ResolvedDatasetConfig": resolved_dataset_config,
            "RequestedTokenizerName": requested_tokenizer_name,
            "ResolvedTokenizerName": resolved_tokenizer_name,
            "UsedFallbackDataset": used_fallback_dataset,
            "Status": f"load_error: {exc}",
        }

    history = train_reasoner_model(
        model=reasoner,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        device=device,
    )

    elapsed_seconds = sum(history.get("epoch_compute_ms", [])) / 1000.0
    if elapsed_seconds <= 0:
        elapsed_seconds = float("nan")

    trainable_params = count_trainable_parameters(reasoner)
    train_tokens_total = int(sum(history.get("epoch_tokens", [])))
    if train_tokens_total <= 0:
        train_tokens_total = int(train_tokens_per_epoch * epochs)

    validation_metrics = evaluate_reasoner_on_real_cases(reasoner, validation_dataloader, device)
    final_loss = float(validation_metrics["RealCaseLoss"])

    sample_inputs, _ = next(iter(validation_dataloader))
    sample_inputs = sample_inputs.to(device_t)

    train_flops = estimate_flops_torch_profiler(
        lambda: reasoner(input_ids=sample_inputs),
        device=device_t,
    )

    compliance = build_compliance_report(
        model_name="SmolLM-135M",
        model=reasoner,
        dataset_name=resolved_dataset_name,
        tokenizer_name=resolved_tokenizer_name,
        device=device_t,
        profiling_backend=profiling_backend,
    )

    row = {
        "Model": "SmolLM-135M",
        "FinalLoss": final_loss,
        "TrainableParams": trainable_params,
        "TrainTimeSeconds": elapsed_seconds,
        "TrainTokens": train_tokens_total,
        "TrainTokensPerSecond": safe_div(train_tokens_total, elapsed_seconds),
        "SecondsPerMParam": safe_div(elapsed_seconds, trainable_params / 1_000_000),
        "TrainFLOPs": train_flops,
        "RealCaseLoss": validation_metrics["RealCaseLoss"],
        "RealCaseAccuracy": validation_metrics["RealCaseAccuracy"],
        "RealCaseEvalSeconds": validation_metrics["RealCaseEvalSeconds"],
        "RequestedDatasetName": requested_dataset_name,
        "RequestedDatasetConfig": requested_dataset_config,
        "ResolvedDatasetName": resolved_dataset_name,
        "ResolvedDatasetConfig": resolved_dataset_config,
        "RequestedTokenizerName": requested_tokenizer_name,
        "ResolvedTokenizerName": resolved_tokenizer_name,
        "UsedFallbackDataset": used_fallback_dataset,
        "Status": "ok",
    }
    row.update(compliance.to_dict())

    print(
        f"[SmolLM-135M] final_loss={final_loss:.6f} "
        f"params={trainable_params:,} time={elapsed_seconds:.2f}s "
        f"tok/s={row['TrainTokensPerSecond']:.2f}"
    )
    if evaluate_real_cases:
        print(
            f"[SmolLM-135M][real] loss={row['RealCaseLoss']:.6f} "
            f"acc={row['RealCaseAccuracy']:.4f} time={row['RealCaseEvalSeconds']:.2f}s"
        )

    return row


def build_models(embed_dim: int) -> Dict[str, torch.nn.Module]:
    return {
        "Transformer": StandardTransformerLayer(embed_dim=embed_dim, num_heads=8, ff_dim=256),
        "DCA": DCA_Layer(embed_dim=embed_dim, sparsity=0.85),
        "MOPN": MOPN_Layer(embed_dim=embed_dim, num_subspaces=4),
        "SCT": SCT_Layer(embed_dim=embed_dim),
        "GMA_MoE": GMA_MoE_Layer(embed_dim=embed_dim, num_experts=4),
    }


def run_benchmark(
    epochs: int = 6,
    lr: float = 1e-3,
    batch_size: int = 24,
    num_samples: int = 8_000,
    seq_len: int = 96,
    embed_dim: int = 64,
    dataset_name: str = CANONICAL_DATASET,
    dataset_config: str = "sample-10BT",
    dataset_split: str = "train",
    tokenizer_name: str = CANONICAL_TOKENIZER,
    val_split: float = 0.05,
    num_proc: Optional[int] = None,
    num_workers: int = 0,
    evaluate_real_cases: bool = True,
    hf_reasoner_model_name: str = "HuggingFaceTB/SmolLM-135M",
    output_csv: str = "benchmark_results.csv",
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    profiling_backend = "cuda_event" if device_t.type == "cuda" else "perf_counter"

    train_loader, val_loader, tokenizer = create_real_dataloaders(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        tokenizer_name=tokenizer_name,
        num_samples=num_samples,
        seq_len=seq_len,
        batch_size=batch_size,
        val_split=val_split,
        num_proc=num_proc,
        num_workers=num_workers,
        pin_memory=device_t.type == "cuda",
        drop_last_train=True,
        seed=42,
    )

    resolved_dataset_name = getattr(train_loader, "_resolved_dataset_name", dataset_name)
    resolved_dataset_config = getattr(train_loader, "_resolved_dataset_config", dataset_config)
    requested_tokenizer_name = getattr(train_loader, "_requested_tokenizer_name", tokenizer_name)
    resolved_tokenizer_name = getattr(train_loader, "_resolved_tokenizer_name", tokenizer_name)
    used_fallback_dataset = bool(getattr(train_loader, "_used_fallback_dataset", False))
    if used_fallback_dataset:
        print(
            "[WARN] Se uso fallback de dataset real: "
            f"requested={dataset_name}({dataset_config}) resolved={resolved_dataset_name}({resolved_dataset_config})"
        )
    if resolved_dataset_name != CANONICAL_DATASET:
        print(
            "[WARN] Dataset resuelto no canonico para ranking oficial: "
            f"resolved={resolved_dataset_name} expected={CANONICAL_DATASET}"
        )
    if resolved_tokenizer_name != CANONICAL_TOKENIZER:
        print(
            "[WARN] Tokenizer resuelto no canonico para ranking oficial: "
            f"resolved={resolved_tokenizer_name} expected={CANONICAL_TOKENIZER}"
        )

    train_tokens_per_epoch = count_tokens_per_epoch(train_loader)
    train_tokens_total = int(train_tokens_per_epoch * epochs)

    eval_loader = val_loader if evaluate_real_cases else None

    models = build_models(embed_dim=embed_dim)

    results = []

    for model_name, model in models.items():
        history = train_model(
            model=model,
            dataloader=train_loader,
            epochs=epochs,
            lr=lr,
            device=device,
        )

        elapsed_seconds = sum(history.get("epoch_compute_ms", [])) / 1000.0
        if elapsed_seconds <= 0:
            elapsed_seconds = float("nan")

        train_tokens_model = int(sum(history.get("epoch_tokens", [])))
        if train_tokens_model <= 0:
            train_tokens_model = train_tokens_total

        trainable_params = count_trainable_parameters(model)

        row = {
            "Model": model_name,
            "FinalLoss": float("nan"),
            "TrainableParams": trainable_params,
            "TrainTimeSeconds": elapsed_seconds,
            "TrainTokens": train_tokens_model,
            "TrainTokensPerSecond": safe_div(train_tokens_model, elapsed_seconds),
            "SecondsPerMParam": safe_div(elapsed_seconds, trainable_params / 1_000_000),
            "TrainFLOPs": float("nan"),
            "RealCaseLoss": float("nan"),
            "RealCaseAccuracy": float("nan"),
            "RealCaseEvalSeconds": float("nan"),
            "RequestedDatasetName": dataset_name,
            "RequestedDatasetConfig": dataset_config,
            "ResolvedDatasetName": resolved_dataset_name,
            "ResolvedDatasetConfig": resolved_dataset_config,
            "RequestedTokenizerName": requested_tokenizer_name,
            "ResolvedTokenizerName": resolved_tokenizer_name,
            "UsedFallbackDataset": used_fallback_dataset,
            "Status": "ok",
        }

        sample_inputs, _ = next(iter(val_loader))
        sample_inputs = sample_inputs.to(device_t)

        row["TrainFLOPs"] = estimate_flops_torch_profiler(
            lambda: model.output_head(model(model.token_embedding(sample_inputs))),
            device=device_t,
        )

        if evaluate_real_cases and eval_loader is not None:
            eval_metrics = evaluate_custom_model_on_real_cases(model, eval_loader, device)
            row.update(eval_metrics)
            row["FinalLoss"] = float(eval_metrics["RealCaseLoss"])
        else:
            row["FinalLoss"] = float(history["loss"][-1]) if history["loss"] else float("nan")

        compliance = build_compliance_report(
            model_name=model_name,
            model=model,
            dataset_name=resolved_dataset_name,
            tokenizer_name=resolved_tokenizer_name,
            device=device_t,
            profiling_backend=profiling_backend,
        )
        row.update(compliance.to_dict())

        results.append(row)

        print(
            f"[{model_name}] final_loss={row['FinalLoss']:.6f} "
            f"params={trainable_params:,} time={elapsed_seconds:.2f}s "
            f"tok/s={row['TrainTokensPerSecond']:.2f}"
        )
        if evaluate_real_cases:
            print(
                f"[{model_name}][real] loss={row['RealCaseLoss']:.6f} "
                f"acc={row['RealCaseAccuracy']:.4f} time={row['RealCaseEvalSeconds']:.2f}s"
            )

    reasoner_result = benchmark_local_reasoner(
        dataloader=train_loader,
        validation_dataloader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        train_tokens_per_epoch=train_tokens_per_epoch,
        requested_dataset_name=dataset_name,
        requested_dataset_config=dataset_config,
        resolved_dataset_name=resolved_dataset_name,
        resolved_dataset_config=resolved_dataset_config,
        requested_tokenizer_name=requested_tokenizer_name,
        resolved_tokenizer_name=resolved_tokenizer_name,
        used_fallback_dataset=used_fallback_dataset,
        evaluate_real_cases=evaluate_real_cases,
        model_name=hf_reasoner_model_name,
    )
    if reasoner_result is not None:
        results.append(reasoner_result)

    results_df = pd.DataFrame(results)
    results_df = add_composite_ranking(results_df)

    ranking_df = results_df.sort_values(
        by="CompositeScore",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    output_path = Path(output_csv)
    results_df.to_csv(output_path, index=False)
    ranking_path = output_path.with_name(f"{output_path.stem}_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)

    report_columns = [
        "CompositeRank",
        "Model",
        "EligibleForRanking",
        "ResolvedDatasetName",
        "ResolvedTokenizerName",
        "UsedFallbackDataset",
        "CompositeScore",
        "QualityScore",
        "EfficiencyScore",
        "FinalLoss",
        "TrainFLOPs",
        "RealCaseAccuracy",
        "TrainTokensPerSecond",
        "TrainTimeSeconds",
    ]
    available_report_columns = [col for col in report_columns if col in ranking_df.columns]
    print("\nTop ranking compuesto (calidad + eficiencia):")
    print(ranking_df[available_report_columns].head(6).to_string(index=False))
    print(f"\nBenchmark exportado a: {output_path.resolve()}")
    print(f"Ranking exportado a: {ranking_path.resolve()}")

    return results_df


if __name__ == "__main__":
    run_benchmark()
