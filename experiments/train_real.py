import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as load_safetensors_file
    from safetensors.torch import save_file as save_safetensors_file
except ImportError:  # pragma: no cover - entorno sin safetensors
    load_safetensors_file = None
    save_safetensors_file = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.real_data import create_real_dataloaders
from neuro_architectures_v2 import NeuroModelV2, entropy_confidence
from utils.compliance import CANONICAL_DATASET, CANONICAL_TOKENIZER, build_compliance_report
from utils.profiler import DeviceTimer


class NeuroModelV2ForLM(torch.nn.Module):
    """
    Adaptador de NeuroModelV2 para entrenamiento autoregresivo con token IDs.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.core = NeuroModelV2(embed_dim=embed_dim, num_classes=vocab_size, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor, exit_threshold: float):
        hidden_states = self.token_embedding(input_ids)
        logits, layers_used, vicarious_loss = self.core(hidden_states, exit_threshold=exit_threshold)
        return logits, layers_used, vicarious_loss


EXPORT_WEIGHTS_FILENAME = "model.safetensors"
EXPORT_CONFIG_FILENAME = "config.json"
EXPORT_TOKENIZER_DIRNAME = "tokenizer"


def select_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        lower_name = name.lower()
        is_bias = name.endswith(".bias")
        is_norm = "norm" in lower_name
        is_embedding = "embedding" in lower_name

        if parameter.ndim < 2 or is_bias or is_norm or is_embedding:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.10,
    min_lr_ratio: float = 0.10,
):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: Dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_model.pt"

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def export_model_bundle(
    export_dir: Path,
    model: NeuroModelV2ForLM,
    tokenizer,
    training_config: Dict,
    checkpoint_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> Path:
    """
    Exporta un artefacto reutilizable para otros desarrollos:
    - model.safetensors
    - config.json (metadatos + hiperparametros)
    - tokenizer/ (si el tokenizer soporta save_pretrained)
    """
    if save_safetensors_file is None:
        raise RuntimeError("safetensors no está instalado. Agrega 'safetensors' al entorno para exportar artefactos.")

    export_dir.mkdir(parents=True, exist_ok=True)

    state_dict_cpu = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    safetensors_metadata = {
        "architecture": "NeuroModelV2ForLM",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_safetensors_file(
        state_dict_cpu,
        str(export_dir / EXPORT_WEIGHTS_FILENAME),
        metadata=safetensors_metadata,
    )

    artifact_config = {
        "format_version": 1,
        "architecture": "NeuroModelV2ForLM",
        "weights_file": EXPORT_WEIGHTS_FILENAME,
        "checkpoint_file": checkpoint_path.name,
        "tokenizer_subdir": EXPORT_TOKENIZER_DIRNAME,
        "model_kwargs": {
            "vocab_size": int(model.vocab_size),
            "embed_dim": int(model.embed_dim),
            "num_layers": int(model.num_layers),
        },
        "best_metrics": {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        },
        "training_config": training_config,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    config_path = export_dir / EXPORT_CONFIG_FILENAME
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact_config, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer_dir = export_dir / EXPORT_TOKENIZER_DIRNAME
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_dir))

    return export_dir


def load_exported_model_bundle(export_dir: Path, device: str = "cpu") -> Tuple[NeuroModelV2ForLM, Dict]:
    """
    Carga un modelo exportado con export_model_bundle(...).
    Devuelve el modelo listo para inferencia y el config del artefacto.
    """
    if load_safetensors_file is None:
        raise RuntimeError("safetensors no está instalado. Agrega 'safetensors' al entorno para cargar artefactos.")

    export_dir = Path(export_dir)
    config_path = export_dir / EXPORT_CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"No existe config de artefacto: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        artifact_config = json.load(handle)

    model_kwargs = artifact_config.get("model_kwargs", {})
    required_model_keys = ("vocab_size", "embed_dim", "num_layers")
    missing_keys = [key for key in required_model_keys if key not in model_kwargs]
    if missing_keys:
        raise ValueError(f"Faltan claves en model_kwargs del artefacto: {missing_keys}")

    model = NeuroModelV2ForLM(
        vocab_size=int(model_kwargs["vocab_size"]),
        embed_dim=int(model_kwargs["embed_dim"]),
        num_layers=int(model_kwargs["num_layers"]),
    )

    weights_filename = artifact_config.get("weights_file", EXPORT_WEIGHTS_FILENAME)
    weights_path = export_dir / weights_filename
    if not weights_path.exists():
        raise FileNotFoundError(f"No existe archivo de pesos exportado: {weights_path}")

    state_dict = load_safetensors_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)

    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()
    return model, artifact_config


def compute_composite_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    layers_used: int,
    num_layers: int,
    vicarious_loss: torch.Tensor,
    pmt_reward_weight: float,
    vicarious_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    prediction_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    depth_ratio = float(layers_used) / float(max(num_layers, 1))
    early_exit_bonus = max(0.0, 1.0 - depth_ratio)
    early_exit_bonus_t = logits.new_tensor(early_exit_bonus)
    confidence_reward = entropy_confidence(logits).mean()
    pmt_reward = early_exit_bonus_t * confidence_reward

    total_loss = (
        prediction_loss
        + vicarious_loss_weight * vicarious_loss
        - pmt_reward_weight * pmt_reward
    )
    return total_loss, prediction_loss, depth_ratio, early_exit_bonus, float(pmt_reward.item())


def format_train_progress(
    epoch: int,
    total_epochs: int,
    batch_idx: int,
    total_batches: int,
    elapsed_seconds: float,
    running_train_loss: float,
) -> str:
    safe_total_batches = max(1, int(total_batches))
    progress = min(1.0, max(0.0, float(batch_idx) / float(safe_total_batches)))

    if 0.0 < progress < 1.0:
        eta_seconds = max(0.0, elapsed_seconds * (1.0 / progress - 1.0))
    else:
        eta_seconds = 0.0

    return (
        f"[epoch {epoch:03d}/{total_epochs:03d}] "
        f"avance={progress * 100.0:5.1f}% "
        f"({batch_idx}/{safe_total_batches}) "
        f"loss={running_train_loss:.4f} "
        f"elapsed={elapsed_seconds:.1f}s eta={eta_seconds:.1f}s"
    )


def training_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    timer: DeviceTimer,
    amp_dtype: torch.dtype,
    use_autocast: bool,
    grad_clip: float,
    exit_threshold: float,
    pmt_reward_weight: float,
    vicarious_loss_weight: float,
) -> Dict[str, float]:
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    step_start = timer.start()

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_autocast,
    ):
        logits, layers_used, vicarious_loss = model(inputs, exit_threshold=exit_threshold)
        loss, prediction_loss, depth_ratio, early_exit_bonus, pmt_reward = compute_composite_loss(
            logits=logits,
            targets=targets,
            layers_used=layers_used,
            num_layers=model.num_layers,
            vicarious_loss=vicarious_loss,
            pmt_reward_weight=pmt_reward_weight,
            vicarious_loss_weight=vicarious_loss_weight,
        )

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    scheduler.step()
    step_ms = timer.stop(step_start)

    token_count = int(targets.numel())
    return {
        "total_loss_sum": float(loss.item()) * token_count,
        "prediction_loss_sum": float(prediction_loss.item()) * token_count,
        "token_count": float(token_count),
        "layers_used": float(layers_used),
        "depth_ratio": float(depth_ratio),
        "early_exit_bonus": float(early_exit_bonus),
        "pmt_reward": float(pmt_reward),
        "step_ms": float(step_ms),
    }


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    timer: DeviceTimer,
    amp_dtype: torch.dtype,
    use_autocast: bool,
    exit_threshold: float,
    pmt_reward_weight: float,
    vicarious_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    # Acumuladores en GPU para evitar sincronización por batch
    total_loss_tensor = torch.tensor(0.0, device=device)
    prediction_loss_tensor = torch.tensor(0.0, device=device)
    layers_used_tensor = torch.tensor(0.0, device=device)
    depth_ratio_tensor = torch.tensor(0.0, device=device)
    early_exit_bonus_tensor = torch.tensor(0.0, device=device)
    pmt_reward_tensor = torch.tensor(0.0, device=device)
    
    total_tokens = 0
    batch_count = 0
    eval_ms_sum = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            eval_start = timer.start()

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_autocast,
            ):
                logits, layers_used, vicarious_loss = model(inputs, exit_threshold=exit_threshold)
                loss, prediction_loss, depth_ratio, early_exit_bonus, pmt_reward = compute_composite_loss(
                    logits=logits,
                    targets=targets,
                    layers_used=layers_used,
                    num_layers=model.num_layers,
                    vicarious_loss=vicarious_loss,
                    pmt_reward_weight=pmt_reward_weight,
                    vicarious_loss_weight=vicarious_loss_weight,
                )

            # Acumulación asíncrona en GPU
            token_count = targets.numel()
            total_loss_tensor += loss * token_count
            prediction_loss_tensor += prediction_loss * token_count
            
            # Estas métricas vienen como floats o tensores escalares de compute_composite_loss
            # Aseguramos que sean tensores para acumular en GPU
            layers_used_tensor += layers_used
            depth_ratio_tensor += depth_ratio
            early_exit_bonus_tensor += early_exit_bonus
            pmt_reward_tensor += pmt_reward
            
            total_tokens += token_count
            batch_count += 1
            
            # Timer sí requiere sincronización puntual, pero es necesario para profiling preciso
            eval_ms_sum += float(timer.stop(eval_start))

    # Sincronización final única
    return {
        "total_loss": float(total_loss_tensor.item()) / max(total_tokens, 1),
        "prediction_loss": float(prediction_loss_tensor.item()) / max(total_tokens, 1),
        "avg_layers_used": float(layers_used_tensor.item()) / max(batch_count, 1),
        "avg_depth_ratio": float(depth_ratio_tensor.item()) / max(batch_count, 1),
        "avg_early_exit_bonus": float(early_exit_bonus_tensor.item()) / max(batch_count, 1),
        "avg_pmt_reward": float(pmt_reward_tensor.item()) / max(batch_count, 1),
        "compute_seconds": eval_ms_sum / 1000.0,
    }


def train(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    pin_memory = device.type == "cuda"

    print(f"Dispositivo: {device}")

    train_loader, val_loader, tokenizer = create_real_dataloaders(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        tokenizer_name=args.tokenizer_name,
        text_column=args.text_column,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_proc=args.num_proc,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last_train=True,
    )

    resolved_dataset_name = getattr(train_loader, "_resolved_dataset_name", args.dataset_name)
    resolved_dataset_config = getattr(train_loader, "_resolved_dataset_config", args.dataset_config)
    requested_tokenizer_name = getattr(train_loader, "_requested_tokenizer_name", args.tokenizer_name)
    resolved_tokenizer_name = getattr(train_loader, "_resolved_tokenizer_name", args.tokenizer_name)
    used_fallback_dataset = bool(getattr(train_loader, "_used_fallback_dataset", False))
    if used_fallback_dataset:
        print(
            "[WARN] Se uso fallback de dataset real: "
            f"requested={args.dataset_name}({args.dataset_config}) "
            f"resolved={resolved_dataset_name}({resolved_dataset_config})"
        )
    if resolved_dataset_name != CANONICAL_DATASET:
        print(
            "[WARN] Dataset resuelto no canonico para cumplimiento R1: "
            f"resolved={resolved_dataset_name} expected={CANONICAL_DATASET}"
        )
    if resolved_tokenizer_name != CANONICAL_TOKENIZER:
        print(
            "[WARN] Tokenizer resuelto no canonico para cumplimiento R1: "
            f"resolved={resolved_tokenizer_name} expected={CANONICAL_TOKENIZER}"
        )

    model = NeuroModelV2ForLM(
        vocab_size=len(tokenizer),
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
    ).to(device)

    profiling_backend = "cuda_event" if device.type == "cuda" else "perf_counter"
    compliance_report = build_compliance_report(
        model_name="NeuroModelV2",
        model=model.core,
        dataset_name=resolved_dataset_name,
        tokenizer_name=resolved_tokenizer_name,
        device=device,
        profiling_backend=profiling_backend,
    )
    print(f"Compliance R1-R5: {compliance_report.to_dict()}")

    optimizer = build_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )

    use_autocast = device.type in {"cuda", "mps"}
    if device.type == "cuda":
        if args.amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    elif device.type == "mps":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
        use_autocast = False

    scaler = None
    if device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    timer = DeviceTimer(device)

    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    export_dir = Path(args.export_dir) if args.export_dir else output_dir / "best_model_export"

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Modelo: NeuroModelV2 | capas={args.num_layers} | params={trainable_params:,} | "
        f"exit_threshold={args.pmt_exit_threshold}"
    )

    total_train_batches = max(1, len(train_loader))
    progress_update_stride = max(1, total_train_batches // 100)
    interactive_stdout = sys.stdout.isatty()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()

        train_total_loss_sum = 0.0
        train_prediction_loss_sum = 0.0
        train_tokens = 0
        train_layers_used_sum = 0.0
        train_depth_ratio_sum = 0.0
        train_early_exit_bonus_sum = 0.0
        train_pmt_reward_sum = 0.0
        train_compute_ms_sum = 0.0
        train_batch_count = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            step_stats = training_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                timer=timer,
                amp_dtype=amp_dtype,
                use_autocast=use_autocast,
                grad_clip=args.grad_clip,
                exit_threshold=args.pmt_exit_threshold,
                pmt_reward_weight=args.pmt_reward_weight,
                vicarious_loss_weight=args.vicarious_loss_weight,
            )
            train_total_loss_sum += step_stats["total_loss_sum"]
            train_prediction_loss_sum += step_stats["prediction_loss_sum"]
            train_tokens += int(step_stats["token_count"])
            train_layers_used_sum += step_stats["layers_used"]
            train_depth_ratio_sum += step_stats["depth_ratio"]
            train_early_exit_bonus_sum += step_stats["early_exit_bonus"]
            train_pmt_reward_sum += step_stats["pmt_reward"]
            train_compute_ms_sum += step_stats["step_ms"]
            train_batch_count += 1

            if (
                batch_idx == 1
                or batch_idx == total_train_batches
                or batch_idx % progress_update_stride == 0
            ):
                running_train_loss = train_total_loss_sum / max(train_tokens, 1)
                elapsed_seconds = time.time() - epoch_start
                progress_line = format_train_progress(
                    epoch=epoch,
                    total_epochs=args.epochs,
                    batch_idx=batch_idx,
                    total_batches=total_train_batches,
                    elapsed_seconds=elapsed_seconds,
                    running_train_loss=running_train_loss,
                )
                if interactive_stdout and batch_idx < total_train_batches:
                    print(progress_line.ljust(140), end="\r", flush=True)
                else:
                    print(progress_line)

        train_loss = train_total_loss_sum / max(train_tokens, 1)
        train_prediction_loss = train_prediction_loss_sum / max(train_tokens, 1)
        train_avg_layers_used = train_layers_used_sum / max(train_batch_count, 1)
        train_avg_depth_ratio = train_depth_ratio_sum / max(train_batch_count, 1)
        train_avg_early_exit_bonus = train_early_exit_bonus_sum / max(train_batch_count, 1)
        train_avg_pmt_reward = train_pmt_reward_sum / max(train_batch_count, 1)

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            timer=timer,
            amp_dtype=amp_dtype,
            use_autocast=use_autocast,
            exit_threshold=args.pmt_exit_threshold,
            pmt_reward_weight=args.pmt_reward_weight,
            vicarious_loss_weight=args.vicarious_loss_weight,
        )
        val_loss = val_metrics["total_loss"]
        val_prediction_loss = val_metrics["prediction_loss"]

        epoch_seconds = time.time() - epoch_start
        train_compute_seconds = train_compute_ms_sum / 1000.0
        train_ppl = math.exp(min(train_prediction_loss, 20.0))
        val_ppl = math.exp(min(val_prediction_loss, 20.0))

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} train_pred={train_prediction_loss:.4f} "
            f"val_loss={val_loss:.4f} val_pred={val_prediction_loss:.4f} "
            f"train_layers={train_avg_layers_used:.2f}/{args.num_layers} "
            f"val_layers={val_metrics['avg_layers_used']:.2f}/{args.num_layers} "
            f"train_bonus={train_avg_early_exit_bonus:.4f} val_bonus={val_metrics['avg_early_exit_bonus']:.4f} "
            f"train_pmt={train_avg_pmt_reward:.4f} val_pmt={val_metrics['avg_pmt_reward']:.4f} "
            f"train_depth={train_avg_depth_ratio:.4f} val_depth={val_metrics['avg_depth_ratio']:.4f} "
            f"train_ppl={train_ppl:.2f} val_ppl={val_ppl:.2f} "
            f"compute_train={train_compute_seconds:.2f}s compute_val={val_metrics['compute_seconds']:.2f}s "
            f"time_wall={epoch_seconds:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_config = {
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "resolved_dataset_name": resolved_dataset_name,
                "resolved_dataset_config": resolved_dataset_config,
                "requested_tokenizer_name": requested_tokenizer_name,
                "resolved_tokenizer_name": resolved_tokenizer_name,
                "used_fallback_dataset": used_fallback_dataset,
                "dataset_split": args.dataset_split,
                "tokenizer_name": args.tokenizer_name,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "embed_dim": args.embed_dim,
                "num_layers": args.num_layers,
                "pmt_exit_threshold": args.pmt_exit_threshold,
                "pmt_reward_weight": args.pmt_reward_weight,
                "vicarious_loss_weight": args.vicarious_loss_weight,
                "profiling_backend": profiling_backend,
                "compliance": compliance_report.to_dict(),
            }
            ckpt_path = save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=checkpoint_config,
            )
            print(f"Nuevo mejor checkpoint guardado en: {ckpt_path.resolve()} | val_loss={val_loss:.4f}")

            exported_path = export_model_bundle(
                export_dir=export_dir,
                model=model,
                tokenizer=tokenizer,
                training_config=checkpoint_config,
                checkpoint_path=ckpt_path,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            print(f"Modelo exportado para reutilizacion en: {exported_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento real de NeuroModelV2 con AMP + AdamW + CosineWarmup + recompensa PMT"
    )

    parser.add_argument("--dataset-name", default=CANONICAL_DATASET)
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--tokenizer-name", default=CANONICAL_TOKENIZER)

    parser.add_argument("--num-samples", type=int, default=200_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=12)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pmt-exit-threshold", type=float, default=0.85)
    parser.add_argument("--pmt-reward-weight", type=float, default=0.05)
    parser.add_argument("--vicarious-loss-weight", type=float, default=0.01)

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Directorio del bundle reutilizable (model.safetensors + config + tokenizer). "
        "Por defecto: <output-dir>/best_model_export",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)
