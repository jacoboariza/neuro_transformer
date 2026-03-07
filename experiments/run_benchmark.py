from pathlib import Path
import math
import re
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

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


SCALING_CONFIGS: Dict[str, Dict[str, int]] = {
    "Micro (6M)": {"embed_dim": 256, "num_layers": 6, "num_heads": 8},
    "Mini (15M)": {"embed_dim": 384, "num_layers": 8, "num_heads": 12},
    "Small (35M)": {"embed_dim": 512, "num_layers": 10, "num_heads": 16},
    "Base (85M)": {"embed_dim": 768, "num_layers": 12, "num_heads": 12},
    "Smol (135M)": {"embed_dim": 896, "num_layers": 14, "num_heads": 14},
}

# Límite de micro-batch por tamaño para prevenir OOM en entrenamientos grandes.
SCALING_MICRO_BATCH_LIMITS: Dict[str, int] = {
    "Micro (6M)": 24,
    "Mini (15M)": 16,
    "Small (35M)": 8,
    "Base (85M)": 4,
    "Smol (135M)": 2,
}

DATASET_METADATA_ATTRIBUTES = (
    "_requested_dataset_name",
    "_requested_dataset_config",
    "_resolved_dataset_name",
    "_resolved_dataset_config",
    "_used_fallback_dataset",
    "_requested_tokenizer_name",
    "_resolved_tokenizer_name",
)


def _size_suffix(size_category: str) -> str:
    return size_category.replace(" ", "")


def is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def resolve_scaling_training_plan(
    size_name: str,
    requested_batch_size: int,
    device_type: str,
) -> Tuple[int, int]:
    if requested_batch_size <= 0:
        raise ValueError("requested_batch_size must be > 0")

    max_micro_batch = SCALING_MICRO_BATCH_LIMITS.get(size_name, requested_batch_size)

    # En CPU mantenemos el tamaño pedido salvo límites explícitos configurados.
    if device_type != "cuda":
        max_micro_batch = min(max_micro_batch, requested_batch_size)

    micro_batch_size = max(1, min(requested_batch_size, max_micro_batch))
    grad_accum_steps = max(1, math.ceil(requested_batch_size / micro_batch_size))
    return micro_batch_size, grad_accum_steps


def build_micro_batch_candidates(initial_micro_batch: int) -> List[int]:
    if initial_micro_batch <= 0:
        raise ValueError("initial_micro_batch must be > 0")

    candidates: List[int] = []
    current = initial_micro_batch
    while True:
        if current not in candidates:
            candidates.append(current)
        if current <= 1:
            break
        current = max(1, current // 2)
    return candidates


def _copy_loader_metadata(source: DataLoader, target: DataLoader) -> None:
    for attr in DATASET_METADATA_ATTRIBUTES:
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


def build_scaled_dataloaders(
    base_train_loader: DataLoader,
    base_val_loader: DataLoader,
    micro_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be > 0")

    train_loader = DataLoader(
        base_train_loader.dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=getattr(base_train_loader, "drop_last", True),
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        base_val_loader.dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=getattr(base_val_loader, "drop_last", False),
    )

    _copy_loader_metadata(base_train_loader, train_loader)
    _copy_loader_metadata(base_val_loader, val_loader)
    return train_loader, val_loader


class StackedArchitectureModel(nn.Module):
    """Apila N bloques de una arquitectura y devuelve hidden states."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        layer_builder: Callable[[], nn.Module],
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([layer_builder() for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden)
        return self.final_norm(hidden)

    def sleep_cycle(self) -> bool:
        executed_sleep_cycle = False
        for layer in self.layers:
            layer_sleep_cycle = getattr(layer, "sleep_cycle", None)
            if callable(layer_sleep_cycle):
                layer_sleep_cycle()
                executed_sleep_cycle = True
        return executed_sleep_cycle


def _attach_model_flags(model: nn.Module, **flags: bool) -> nn.Module:
    for name, value in flags.items():
        setattr(model, name, value)
    return model


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


def _probe_cuda_runtime() -> None:
    """Valida CUDA ejecutando un kernel de embedding y sincronizando el stream."""
    probe_embedding = nn.Embedding(8, 4, device="cuda")
    probe_input = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device="cuda")
    _ = probe_embedding(probe_input)
    torch.cuda.synchronize()


def select_benchmark_device() -> Tuple[str, Optional[str]]:
    """
    Selecciona dispositivo para benchmark con fallback seguro a CPU cuando
    CUDA está visible pero no es compatible con la GPU/build actual.
    """
    if not torch.cuda.is_available():
        return "cpu", "CUDA no está disponible en este entorno."

    try:
        capability = torch.cuda.get_device_capability(0)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        supported_arches = torch.cuda.get_arch_list()

        if supported_arches and required_arch not in supported_arches:
            supported_msg = " ".join(supported_arches)
            return (
                "cpu",
                "GPU detectada con CUDA capability "
                f"{required_arch}, pero el build actual de PyTorch solo soporta: {supported_msg}.",
            )

        _probe_cuda_runtime()
        return "cuda", None
    except Exception as exc:  # pragma: no cover - depende de entorno CUDA
        return "cpu", f"CUDA detectada, pero no utilizable con la instalación actual: {exc}"


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

    # Acumuladores en GPU
    total_loss_tensor = torch.tensor(0.0, device=device_t)
    total_correct_tensor = torch.tensor(0.0, device=device_t)
    
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

            # Acumulación en GPU
            total_loss_tensor += loss
            total_correct_tensor += (predictions == targets).sum()
            
            # Métricas de conteo (CPU es rápido para esto si no depende de GPU)
            # targets.numel() es conocido por el shape en CPU si el dataloader funciona bien,
            # pero targets está en GPU. targets.size() devuelve torch.Size que está en CPU.
            total_tokens += int(targets.numel())
            batch_count += 1
            
            # Timer requiere sincronización puntual
            total_eval_ms += timer.stop(eval_start)

    elapsed = total_eval_ms / 1000.0
    
    # Sincronización final
    avg_loss = float(total_loss_tensor.item()) / max(batch_count, 1)
    accuracy = float(total_correct_tensor.item()) / max(total_tokens, 1)

    return {
        "RealCaseLoss": avg_loss,
        "RealCaseAccuracy": accuracy,
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

    # Acumuladores en GPU
    total_loss_tensor = torch.tensor(0.0, device=device_t)
    total_correct_tensor = torch.tensor(0.0, device=device_t)
    
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

            # Acumulación en GPU
            total_loss_tensor += loss
            total_correct_tensor += (predictions == targets).sum()
            
            total_tokens += int(targets.numel())
            batch_count += 1
            
            # Timer requiere sincronización puntual
            total_eval_ms += timer.stop(eval_start)

    elapsed = total_eval_ms / 1000.0
    
    # Sincronización final
    avg_loss = float(total_loss_tensor.item()) / max(batch_count, 1)
    accuracy = float(total_correct_tensor.item()) / max(total_tokens, 1)

    return {
        "RealCaseLoss": avg_loss,
        "RealCaseAccuracy": accuracy,
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
        # Acumuladores en GPU
        total_loss_tensor = torch.tensor(0.0, device=device_t)
        
        batch_count = 0
        total_tokens = 0
        
        # Medimos el epoch completo para throughput real
        epoch_start_marker = timer.start()

        for inputs, targets in dataloader:
            inputs = inputs.to(device_t, non_blocking=True)
            targets = targets.to(device_t, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs)
            logits = outputs.logits

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            # Acumulación asíncrona
            total_loss_tensor += loss
            
            total_tokens += int(targets.numel())
            batch_count += 1
        
        # Sincronización única al final del epoch
        epoch_ms = timer.stop(epoch_start_marker)
        
        epoch_loss = float(total_loss_tensor.item()) / max(batch_count, 1)
        avg_step_ms = epoch_ms / max(batch_count, 1)

        history["loss"].append(epoch_loss)
        history["avg_step_ms"].append(avg_step_ms)
        history["epoch_tokens"].append(total_tokens)
        history["epoch_compute_ms"].append(epoch_ms)

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
    size_category: str = "Reference (135M)",
) -> Optional[dict]:
    device_t = torch.device(device)
    profiling_backend = "cuda_event" if device_t.type == "cuda" else "perf_counter"

    try:
        _tokenizer, reasoner = load_local_reasoner(model_name=model_name, device=device)
    except Exception as exc:
        print(f"[SmolLM] No se pudo cargar {model_name}: {exc}")
        return {
            "Model": "SmolLM-135M",
            "SizeCategory": size_category,
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
        "SizeCategory": size_category,
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


def build_models(
    embed_dim: int,
    num_layers: int,
    num_heads: int,
) -> Dict[str, Callable[[], torch.nn.Module]]:
    ff_dim = embed_dim * 4

    return {
        "Transformer": lambda: _attach_model_flags(
            StackedArchitectureModel(
                embed_dim=embed_dim,
                num_layers=num_layers,
                layer_builder=lambda: StandardTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                ),
            ),
            uses_sparse_dca=False,
        ),
        "DCA": lambda: _attach_model_flags(
            StackedArchitectureModel(
                embed_dim=embed_dim,
                num_layers=num_layers,
                layer_builder=lambda: DCA_Layer(embed_dim=embed_dim, sparsity=0.85),
            ),
            uses_sparse_dca=True,
        ),
        "MOPN": lambda: _attach_model_flags(
            StackedArchitectureModel(
                embed_dim=embed_dim,
                num_layers=num_layers,
                layer_builder=lambda: MOPN_Layer(embed_dim=embed_dim, num_subspaces=4),
            ),
            uses_sparse_dca=False,
        ),
        "SCT": lambda: _attach_model_flags(
            StackedArchitectureModel(
                embed_dim=embed_dim,
                num_layers=num_layers,
                layer_builder=lambda: SCT_Layer(embed_dim=embed_dim),
            ),
            uses_sparse_dca=False,
        ),
        "GMA_MoE": lambda: _attach_model_flags(
            StackedArchitectureModel(
                embed_dim=embed_dim,
                num_layers=num_layers,
                layer_builder=lambda: GMA_MoE_Layer(embed_dim=embed_dim, num_experts=4),
            ),
            uses_sparse_dca=False,
        ),
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
    scaling_configs: Optional[Dict[str, Dict[str, int]]] = None,
) -> pd.DataFrame:
    device, device_reason = select_benchmark_device()
    print(f"[{'ADVERTENCIA' if device == 'cpu' else 'INFO'}] Ejecutando benchmark en dispositivo: {device.upper()}")
    if device_reason:
        print(f"  -> {device_reason}")
    if device == "cpu":
        print("  -> La ejecución en CPU será significativamente más lenta. Considere instalar PyTorch con soporte CUDA.")
    
    device_t = torch.device(device)
    profiling_backend = "cuda_event" if device_t.type == "cuda" else "perf_counter"

    base_train_loader, base_val_loader, _tokenizer = create_real_dataloaders(
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

    resolved_dataset_name = getattr(base_train_loader, "_resolved_dataset_name", dataset_name)
    resolved_dataset_config = getattr(base_train_loader, "_resolved_dataset_config", dataset_config)
    requested_tokenizer_name = getattr(base_train_loader, "_requested_tokenizer_name", tokenizer_name)
    resolved_tokenizer_name = getattr(base_train_loader, "_resolved_tokenizer_name", tokenizer_name)
    used_fallback_dataset = bool(getattr(base_train_loader, "_used_fallback_dataset", False))
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

    resolved_scaling_configs = scaling_configs if scaling_configs is not None else SCALING_CONFIGS
    if not resolved_scaling_configs:
        resolved_scaling_configs = {
            "Single": {
                "embed_dim": embed_dim,
                "num_layers": 6,
                "num_heads": 8 if embed_dim % 8 == 0 else 4,
            }
        }

    results: List[Dict[str, object]] = []

    for size_name, size_config in resolved_scaling_configs.items():
        size_embed_dim = int(size_config["embed_dim"])
        size_num_layers = int(size_config["num_layers"])
        size_num_heads = int(size_config["num_heads"])

        planned_micro_batch, planned_grad_accum = resolve_scaling_training_plan(
            size_name=size_name,
            requested_batch_size=batch_size,
            device_type=device_t.type,
        )
        micro_batch_candidates = build_micro_batch_candidates(planned_micro_batch)

        print(
            f"\n[SCALING] {size_name} -> embed_dim={size_embed_dim}, "
            f"layers={size_num_layers}, heads={size_num_heads}, target_batch={batch_size}, "
            f"micro_batch_inicial={planned_micro_batch}, grad_accum_inicial={planned_grad_accum}"
        )

        model_factories = build_models(
            embed_dim=size_embed_dim,
            num_layers=size_num_layers,
            num_heads=size_num_heads,
        )

        for base_model_name, model_factory in model_factories.items():
            model_label = f"{base_model_name}_{_size_suffix(size_name)}"
            trained_model: Optional[torch.nn.Module] = None
            history: Optional[Dict[str, list]] = None
            effective_train_loader: Optional[DataLoader] = None
            effective_eval_loader: Optional[DataLoader] = None
            used_micro_batch = planned_micro_batch
            used_grad_accum = planned_grad_accum
            train_status = "ok"
            last_exception: Optional[BaseException] = None

            for micro_batch_candidate in micro_batch_candidates:
                grad_accum_steps = max(1, math.ceil(batch_size / micro_batch_candidate))
                used_micro_batch = micro_batch_candidate
                used_grad_accum = grad_accum_steps
                model_train_loader, model_val_loader = build_scaled_dataloaders(
                    base_train_loader=base_train_loader,
                    base_val_loader=base_val_loader,
                    micro_batch_size=micro_batch_candidate,
                    num_workers=num_workers,
                    pin_memory=device_t.type == "cuda",
                    seed=42,
                )

                try:
                    candidate_model = model_factory()
                    candidate_history = train_model(
                        model=candidate_model,
                        dataloader=model_train_loader,
                        epochs=epochs,
                        lr=lr,
                        device=device,
                        grad_accum_steps=grad_accum_steps,
                    )
                    trained_model = candidate_model
                    history = candidate_history
                    effective_train_loader = model_train_loader
                    effective_eval_loader = model_val_loader
                    used_micro_batch = micro_batch_candidate
                    used_grad_accum = grad_accum_steps
                    break
                except RuntimeError as exc:
                    last_exception = exc
                    if device_t.type == "cuda" and is_cuda_oom_error(exc) and micro_batch_candidate > 1:
                        next_micro_batch = max(1, micro_batch_candidate // 2)
                        print(
                            f"[WARN][{model_label}] CUDA OOM con micro_batch={micro_batch_candidate}. "
                            f"Reintentando con micro_batch={next_micro_batch}."
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    train_status = f"train_error: {exc}"
                    break

            if trained_model is None or history is None or effective_train_loader is None or effective_eval_loader is None:
                if train_status == "ok":
                    train_status = f"train_error: {last_exception}" if last_exception else "train_error: unknown"

                row = {
                    "Model": model_label,
                    "SizeCategory": size_name,
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
                    "MicroBatchSize": used_micro_batch,
                    "GradAccumSteps": used_grad_accum,
                    "EffectiveBatchSize": used_micro_batch * used_grad_accum,
                    "R1_RealData": False,
                    "R2_SparseDCA": False,
                    "R3_TokenMaskingPMT": False,
                    "R4_VLMDetach": False,
                    "R5_PreciseProfiling": False,
                    "EligibleForRanking": False,
                    "RequestedDatasetName": dataset_name,
                    "RequestedDatasetConfig": dataset_config,
                    "ResolvedDatasetName": resolved_dataset_name,
                    "ResolvedDatasetConfig": resolved_dataset_config,
                    "RequestedTokenizerName": requested_tokenizer_name,
                    "ResolvedTokenizerName": resolved_tokenizer_name,
                    "UsedFallbackDataset": used_fallback_dataset,
                    "Status": train_status,
                }
                results.append(row)

                try:
                    pd.DataFrame(results).to_csv(output_csv, index=False)
                except Exception as exc:
                    print(f"[WARN] No se pudo guardar CSV parcial: {exc}")

                print(f"[ERROR][{model_label}] {train_status}")
                continue

            train_tokens_per_epoch = count_tokens_per_epoch(effective_train_loader)
            train_tokens_total = int(train_tokens_per_epoch * epochs)

            elapsed_seconds = sum(history.get("epoch_compute_ms", [])) / 1000.0
            if elapsed_seconds <= 0:
                elapsed_seconds = float("nan")

            train_tokens_model = int(sum(history.get("epoch_tokens", [])))
            if train_tokens_model <= 0:
                train_tokens_model = train_tokens_total

            trainable_params = count_trainable_parameters(trained_model)

            row = {
                "Model": model_label,
                "SizeCategory": size_name,
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
                "MicroBatchSize": used_micro_batch,
                "GradAccumSteps": used_grad_accum,
                "EffectiveBatchSize": used_micro_batch * used_grad_accum,
                "RequestedDatasetName": dataset_name,
                "RequestedDatasetConfig": dataset_config,
                "ResolvedDatasetName": resolved_dataset_name,
                "ResolvedDatasetConfig": resolved_dataset_config,
                "RequestedTokenizerName": requested_tokenizer_name,
                "ResolvedTokenizerName": resolved_tokenizer_name,
                "UsedFallbackDataset": used_fallback_dataset,
                "Status": "ok",
            }

            sample_inputs, _ = next(iter(effective_eval_loader))
            sample_inputs = sample_inputs.to(device_t)

            try:
                row["TrainFLOPs"] = estimate_flops_torch_profiler(
                    lambda: trained_model.output_head(trained_model(trained_model.token_embedding(sample_inputs))),
                    device=device_t,
                )
            except Exception as exc:
                row["TrainFLOPs"] = float("nan")
                row["Status"] = f"profile_error: {exc}"

            if evaluate_real_cases:
                try:
                    eval_metrics = evaluate_custom_model_on_real_cases(trained_model, effective_eval_loader, device)
                    row.update(eval_metrics)
                    row["FinalLoss"] = float(eval_metrics["RealCaseLoss"])
                except Exception as exc:
                    row["Status"] = f"eval_error: {exc}" if row["Status"] == "ok" else f"{row['Status']} | eval_error: {exc}"
            else:
                row["FinalLoss"] = float(history["loss"][-1]) if history["loss"] else float("nan")

            compliance = build_compliance_report(
                model_name=base_model_name,
                model=trained_model,
                dataset_name=resolved_dataset_name,
                tokenizer_name=resolved_tokenizer_name,
                device=device_t,
                profiling_backend=profiling_backend,
            )
            row.update(compliance.to_dict())

            results.append(row)

            try:
                pd.DataFrame(results).to_csv(output_csv, index=False)
            except Exception as exc:
                print(f"[WARN] No se pudo guardar CSV parcial: {exc}")

            print(
                f"[{model_label}] final_loss={row['FinalLoss']:.6f} "
                f"params={trainable_params:,} time={elapsed_seconds:.2f}s "
                f"tok/s={row['TrainTokensPerSecond']:.2f} "
                f"micro_batch={used_micro_batch} grad_accum={used_grad_accum}"
            )
            if evaluate_real_cases:
                print(
                    f"[{model_label}][real] loss={row['RealCaseLoss']:.6f} "
                    f"acc={row['RealCaseAccuracy']:.4f} time={row['RealCaseEvalSeconds']:.2f}s"
                )

            if device_t.type == "cuda":
                torch.cuda.empty_cache()

    reasoner_size_category = "Reference (135M)"
    reasoner_micro_batch, _ = resolve_scaling_training_plan(
        size_name="Smol (135M)",
        requested_batch_size=batch_size,
        device_type=device_t.type,
    )
    reasoner_result: Optional[Dict[str, object]] = None

    for reasoner_micro_batch_candidate in build_micro_batch_candidates(reasoner_micro_batch):
        reasoner_train_loader, reasoner_val_loader = build_scaled_dataloaders(
            base_train_loader=base_train_loader,
            base_val_loader=base_val_loader,
            micro_batch_size=reasoner_micro_batch_candidate,
            num_workers=num_workers,
            pin_memory=device_t.type == "cuda",
            seed=42,
        )
        reasoner_train_tokens_per_epoch = count_tokens_per_epoch(reasoner_train_loader)

        try:
            reasoner_result = benchmark_local_reasoner(
                dataloader=reasoner_train_loader,
                validation_dataloader=reasoner_val_loader,
                epochs=epochs,
                lr=lr,
                device=device,
                train_tokens_per_epoch=reasoner_train_tokens_per_epoch,
                requested_dataset_name=dataset_name,
                requested_dataset_config=dataset_config,
                resolved_dataset_name=resolved_dataset_name,
                resolved_dataset_config=resolved_dataset_config,
                requested_tokenizer_name=requested_tokenizer_name,
                resolved_tokenizer_name=resolved_tokenizer_name,
                used_fallback_dataset=used_fallback_dataset,
                evaluate_real_cases=evaluate_real_cases,
                model_name=hf_reasoner_model_name,
                size_category=reasoner_size_category,
            )

            if reasoner_result is not None:
                reasoner_result["MicroBatchSize"] = reasoner_micro_batch_candidate
                reasoner_result["GradAccumSteps"] = 1
                reasoner_result["EffectiveBatchSize"] = reasoner_micro_batch_candidate
            break
        except RuntimeError as exc:
            if device_t.type == "cuda" and is_cuda_oom_error(exc) and reasoner_micro_batch_candidate > 1:
                next_reasoner_micro_batch = max(1, reasoner_micro_batch_candidate // 2)
                print(
                    "[WARN][SmolLM-135M] CUDA OOM con "
                    f"micro_batch={reasoner_micro_batch_candidate}. Reintentando con micro_batch={next_reasoner_micro_batch}."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            reasoner_result = {
                "Model": "SmolLM-135M",
                "SizeCategory": reasoner_size_category,
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
                "MicroBatchSize": reasoner_micro_batch_candidate,
                "GradAccumSteps": 1,
                "EffectiveBatchSize": reasoner_micro_batch_candidate,
                "R1_RealData": False,
                "R2_SparseDCA": False,
                "R3_TokenMaskingPMT": False,
                "R4_VLMDetach": False,
                "R5_PreciseProfiling": False,
                "EligibleForRanking": False,
                "RequestedDatasetName": dataset_name,
                "RequestedDatasetConfig": dataset_config,
                "ResolvedDatasetName": resolved_dataset_name,
                "ResolvedDatasetConfig": resolved_dataset_config,
                "RequestedTokenizerName": requested_tokenizer_name,
                "ResolvedTokenizerName": resolved_tokenizer_name,
                "UsedFallbackDataset": used_fallback_dataset,
                "Status": f"train_error: {exc}",
            }
            break

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
        "SizeCategory",
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
    import argparse

    parser = argparse.ArgumentParser(description="Ejecutar benchmark de arquitecturas neuro-inspiradas.")
    parser.add_argument("--epochs", type=int, default=6, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=24, help="Tamaño del batch")
    parser.add_argument("--num-samples", type=int, default=8_000, help="Número de muestras del dataset")
    parser.add_argument("--seq-len", type=int, default=96, help="Longitud de la secuencia")
    parser.add_argument("--embed-dim", type=int, default=64, help="Dimensión del embedding")
    parser.add_argument("--num-workers", type=int, default=0, help="Número de workers para DataLoader")
    parser.add_argument("--dataset-name", type=str, default=CANONICAL_DATASET, help="Nombre del dataset")
    parser.add_argument("--dataset-config", type=str, default="sample-10BT", help="Configuración del dataset")
    parser.add_argument("--tokenizer-name", type=str, default=CANONICAL_TOKENIZER, help="Nombre del tokenizer")
    parser.add_argument("--no-real-cases", action="store_true", help="Desactivar evaluación en casos reales")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv", help="Archivo de salida CSV")

    args = parser.parse_args()

    run_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_workers=args.num_workers,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer_name,
        evaluate_real_cases=not args.no_real_cases,
        output_csv=args.output_csv,
    )
