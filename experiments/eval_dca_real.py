import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.real_data import create_real_dataloaders
from experiments.train_real import evaluate, load_exported_model_bundle
from utils.profiler import DeviceTimer


def _format_sm_arch(capability: Tuple[int, int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def _probe_cuda_runtime() -> Optional[str]:
    """
    Verifica que CUDA no solo este visible, sino realmente utilizable.
    """
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() == False"

    try:
        capability = torch.cuda.get_device_capability(0)
        required_arch = _format_sm_arch(capability)
        supported_arches = torch.cuda.get_arch_list()
        if supported_arches and required_arch not in supported_arches:
            supported_msg = " ".join(supported_arches)
            return (
                "GPU detectada con capability "
                f"{required_arch}, pero el build actual de PyTorch soporta: {supported_msg}"
            )

        probe_embedding = torch.nn.Embedding(8, 4, device="cuda")
        probe_input = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device="cuda")
        _ = probe_embedding(probe_input)
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - depende de entorno
        return str(exc)

    return None


def select_eval_device(preferred: str = "auto", require_cuda: bool = False) -> torch.device:
    preferred = preferred.lower()
    if preferred not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Dispositivo no soportado: {preferred}")

    if preferred == "cpu":
        return torch.device("cpu")

    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS no esta disponible en este entorno.")

    if preferred == "cuda":
        cuda_error = _probe_cuda_runtime()
        if cuda_error is not None:
            raise RuntimeError(f"CUDA solicitado pero no usable: {cuda_error}")
        return torch.device("cuda")

    cuda_error = _probe_cuda_runtime()
    if cuda_error is None:
        return torch.device("cuda")

    if require_cuda:
        raise RuntimeError(f"--require-cuda activo y CUDA no usable: {cuda_error}")

    if torch.backends.mps.is_available():
        print(f"[WARN] CUDA no usable, se usara MPS: {cuda_error}")
        return torch.device("mps")

    print(f"[WARN] CUDA no usable, fallback a CPU: {cuda_error}")
    return torch.device("cpu")


def configure_cuda_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_amp_dtype(device: torch.device, requested_amp_dtype: str) -> torch.dtype:
    requested_amp_dtype = requested_amp_dtype.lower()

    if device.type == "cuda":
        if requested_amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            print("[WARN] bf16 no soportado por la GPU actual, usando fp16.")
            return torch.float16
        if requested_amp_dtype == "fp16":
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if device.type == "mps":
        return torch.float16

    return torch.float32


def count_eval_tokens(dataloader) -> int:
    dataset = getattr(dataloader, "dataset", None)
    if isinstance(dataset, torch.utils.data.TensorDataset):
        if len(dataset.tensors) >= 2:
            return int(dataset.tensors[1].numel())

    total_tokens = 0
    for _, targets in dataloader:
        total_tokens += int(targets.numel())
    return total_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida un bundle exportado de DCA en datos reales y prioriza ejecucion en CUDA "
            "cuando este disponible."
        )
    )

    parser.add_argument("--export-dir", default="checkpoints_dca/best_model_export")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pmt-exit-threshold", type=float, default=None)
    parser.add_argument("--pmt-reward-weight", type=float, default=None)
    parser.add_argument("--vicarious-loss-weight", type=float, default=None)
    parser.add_argument("--output-json", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_dir = Path(args.export_dir)
    device = select_eval_device(preferred=args.device, require_cuda=args.require_cuda)
    configure_cuda_backend(device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA runtime: {torch.version.cuda}")
    print(f"Dispositivo de evaluacion: {device}")

    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        gpu_capability = _format_sm_arch(torch.cuda.get_device_capability(gpu_index))
        print(f"GPU activa: {gpu_name} ({gpu_capability})")

    model, artifact_config = load_exported_model_bundle(export_dir=export_dir, device=device.type)
    training_config = artifact_config.get("training_config", {})

    tokenizer_name = (
        args.tokenizer_name
        or training_config.get("tokenizer_name")
        or training_config.get("resolved_tokenizer_name")
        or "HuggingFaceTB/SmolLM-135M"
    )
    seq_len = int(args.seq_len if args.seq_len is not None else training_config.get("seq_len", 512))

    pmt_exit_threshold = (
        float(args.pmt_exit_threshold)
        if args.pmt_exit_threshold is not None
        else float(training_config.get("pmt_exit_threshold", 0.85))
    )
    pmt_reward_weight = (
        float(args.pmt_reward_weight)
        if args.pmt_reward_weight is not None
        else float(training_config.get("pmt_reward_weight", 0.05))
    )
    vicarious_loss_weight = (
        float(args.vicarious_loss_weight)
        if args.vicarious_loss_weight is not None
        else float(training_config.get("vicarious_loss_weight", 0.01))
    )

    _, val_loader, _tokenizer = create_real_dataloaders(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        tokenizer_name=tokenizer_name,
        text_column=args.text_column,
        num_samples=args.num_samples,
        seq_len=seq_len,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_proc=args.num_proc,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last_train=False,
    )

    amp_dtype = resolve_amp_dtype(device=device, requested_amp_dtype=args.amp_dtype)
    use_autocast = device.type in {"cuda", "mps"}

    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        timer=DeviceTimer(device),
        amp_dtype=amp_dtype,
        use_autocast=use_autocast,
        exit_threshold=pmt_exit_threshold,
        pmt_reward_weight=pmt_reward_weight,
        vicarious_loss_weight=vicarious_loss_weight,
    )

    eval_tokens = count_eval_tokens(val_loader)
    compute_seconds = float(metrics.get("compute_seconds", 0.0))
    tokens_per_second = eval_tokens / compute_seconds if compute_seconds > 0 else float("nan")

    val_pred_loss = float(metrics["prediction_loss"])
    metrics["val_ppl"] = math.exp(min(val_pred_loss, 20.0))
    metrics["eval_tokens"] = eval_tokens
    metrics["eval_tokens_per_second"] = tokens_per_second

    output_payload: Dict[str, object] = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_dir": str(export_dir.resolve()),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "evaluation_config": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "tokenizer_name": tokenizer_name,
            "num_samples": int(args.num_samples),
            "seq_len": int(seq_len),
            "batch_size": int(args.batch_size),
            "val_split": float(args.val_split),
            "pmt_exit_threshold": pmt_exit_threshold,
            "pmt_reward_weight": pmt_reward_weight,
            "vicarious_loss_weight": vicarious_loss_weight,
        },
        "artifact_best_metrics": artifact_config.get("best_metrics", {}),
        "metrics": metrics,
    }

    output_path = Path(args.output_json) if args.output_json else export_dir / "eval_real_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(
        "[OK] "
        f"val_loss={metrics['total_loss']:.4f} "
        f"val_pred={metrics['prediction_loss']:.4f} "
        f"ppl={metrics['val_ppl']:.2f} "
        f"tok/s={metrics['eval_tokens_per_second']:.2f}"
    )
    print(f"Reporte JSON guardado en: {output_path.resolve()}")


if __name__ == "__main__":
    main()
