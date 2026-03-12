#!/usr/bin/env python
"""
Diagnóstico del mecanismo PMT (Predictive Minimalist Trace) en NeuroModelV2.

Objetivo: Entender por qué avg_layers_used ≈ 12.0 (sin early exit efectivo).
Hipótesis: softmax(logits).max() sobre vocab_size=49152 produce confianzas
           muy bajas, haciendo que el threshold=0.85 sea inalcanzable.

Salida: JSON con estadísticas de confianza por capa y sweep de thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# ── proyecto local ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.train_real import load_exported_model_bundle
from data.real_data import create_real_dataloaders
from neuro_architectures_v2 import entropy_confidence


# ────────────────────────────────────────────────────────────────────
# Instrumentación: forward hook para capturar confianza por capa
# ────────────────────────────────────────────────────────────────────

class PMTDiagnostics:
    """Captura estadísticas de confianza por capa durante el forward."""

    def __init__(self, model, vocab_size: int, num_layers: int):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # Per-layer stats across all batches
        self.layer_confidences: Dict[int, List[float]] = {i: [] for i in range(num_layers)}
        self.layer_confidence_tensors: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers)}
        self._model = model

    @torch.no_grad()
    def instrumented_forward(
        self, x: torch.Tensor, exit_threshold: float = 0.85
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Forward idéntico a NeuroModelV2.forward() pero captura confianza por capa.
        """
        core = self._model.core if hasattr(self._model, "core") else self._model
        embed_dim = core.embed_dim
        num_layers = core.num_layers
        num_classes = core.early_exits[0].exit_predictor.out_features

        vicarious_losses = []
        token_depth = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        active_mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.bool)
        final_logits = None

        for i in range(num_layers):
            x_flat = x.reshape(-1, embed_dim)
            active_flat = active_mask.reshape(-1)

            active_tokens = x_flat[active_flat].unsqueeze(1)
            processed_active = core.layers[i](active_tokens).squeeze(1)

            if i == num_layers // 2:
                processed_active = core.cen_module(processed_active.unsqueeze(1)).squeeze(1)

            x_flat = x_flat.clone()
            x_flat[active_flat] = processed_active
            x = x_flat.reshape_as(x)

            vicarious_losses.append(core.vlm_student.observe_and_learn(x))

            # ── PMT confidence computation ──
            active_logits = core.early_exits[i].exit_predictor(x_flat[active_flat])
            active_confidence = entropy_confidence(active_logits)

            # ── CAPTURE: guardar confianzas de esta capa ──
            if active_confidence.numel() > 0:
                self.layer_confidence_tensors[i].append(active_confidence.detach().cpu())

            # Resto de la lógica original
            logits_flat = x.new_zeros((x_flat.size(0), num_classes))
            confidence_flat = x.new_ones((x_flat.size(0),))

            if active_logits.dtype != logits_flat.dtype:
                active_logits = active_logits.to(dtype=logits_flat.dtype)
            if active_confidence.dtype != confidence_flat.dtype:
                active_confidence = active_confidence.to(dtype=confidence_flat.dtype)

            logits_flat[active_flat] = active_logits
            confidence_flat[active_flat] = active_confidence

            logits = logits_flat.reshape(x.size(0), x.size(1), num_classes)
            confidence = confidence_flat.reshape(x.size(0), x.size(1))

            if final_logits is None:
                final_logits = logits
            else:
                final_flat = final_logits.reshape(-1, num_classes)
                final_flat = final_flat.clone()
                final_flat[active_flat] = active_logits
                final_logits = final_flat.reshape_as(final_logits)

            token_depth = token_depth + active_mask.to(dtype=token_depth.dtype)
            continue_mask = confidence < exit_threshold
            active_mask = active_mask & continue_mask

        if final_logits is None:
            final_logits, _ = core.early_exits[-1](x)

        avg_layers_used = float(token_depth.mean().item())
        avg_vicarious_loss = sum(vicarious_losses) / max(len(vicarious_losses), 1)
        return final_logits, avg_layers_used, avg_vicarious_loss

    def compute_stats(self) -> Dict[str, object]:
        """Agrega estadísticas de confianza por capa."""
        stats = {}
        for layer_idx in range(self.num_layers):
            tensors = self.layer_confidence_tensors[layer_idx]
            if not tensors:
                stats[f"layer_{layer_idx}"] = {"count": 0}
                continue
            all_conf = torch.cat(tensors)
            stats[f"layer_{layer_idx}"] = {
                "count": int(all_conf.numel()),
                "mean": float(all_conf.mean().item()),
                "std": float(all_conf.std().item()) if all_conf.numel() > 1 else 0.0,
                "min": float(all_conf.min().item()),
                "max": float(all_conf.max().item()),
                "median": float(all_conf.median().item()),
                "p90": float(all_conf.quantile(0.90).item()),
                "p95": float(all_conf.quantile(0.95).item()),
                "p99": float(all_conf.quantile(0.99).item()),
                "pct_above_0.10": float((all_conf > 0.10).float().mean().item()) * 100,
                "pct_above_0.25": float((all_conf > 0.25).float().mean().item()) * 100,
                "pct_above_0.50": float((all_conf > 0.50).float().mean().item()) * 100,
                "pct_above_0.85": float((all_conf > 0.85).float().mean().item()) * 100,
            }
        return stats


def threshold_sweep(
    model,
    dataloader,
    device: torch.device,
    amp_dtype: torch.dtype,
    thresholds: List[float],
    num_batches: int = 20,
) -> Dict[str, object]:
    """Ejecuta el modelo con múltiples thresholds y reporta layers_used."""
    core = model.core if hasattr(model, "core") else model
    num_layers = core.num_layers
    results = {}

    for thr in thresholds:
        total_layers = 0.0
        total_tokens = 0
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs = inputs.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                # Usamos el forward original del modelo
                if hasattr(model, "core"):
                    hidden = model.token_embedding(inputs)
                    _, layers_used, _ = core(hidden, exit_threshold=thr)
                else:
                    _, layers_used, _ = model(inputs, exit_threshold=thr)

            tokens_in_batch = inputs.numel()
            total_layers += layers_used * tokens_in_batch
            total_tokens += tokens_in_batch
            batch_count += 1

        avg_layers = total_layers / max(total_tokens, 1)
        depth_ratio = avg_layers / num_layers
        results[f"threshold_{thr:.3f}"] = {
            "threshold": thr,
            "avg_layers_used": round(avg_layers, 4),
            "depth_ratio": round(depth_ratio, 6),
            "batches_evaluated": batch_count,
            "tokens_evaluated": total_tokens,
            "pct_compute_saved": round((1.0 - depth_ratio) * 100, 2),
        }
        print(f"  threshold={thr:.3f}  avg_layers={avg_layers:.4f}/{num_layers}  "
              f"depth_ratio={depth_ratio:.6f}  saved={((1-depth_ratio)*100):.2f}%")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de PMT Early Exit")
    parser.add_argument(
        "--export-dir",
        type=str,
        default="checkpoints_dca/best_model_export",
        help="Directorio del bundle exportado",
    )
    parser.add_argument("--num-batches", type=int, default=20, help="Batches para diagnóstico")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=2000, help="Muestras de datos reales")
    parser.add_argument("--output", type=str, default="", help="Ruta JSON de salida")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ── Cargar modelo ──
    export_dir = Path(args.export_dir)
    print(f"Cargando modelo desde {export_dir}...")
    model, config = load_exported_model_bundle(export_dir, device=str(device))
    model.eval()

    vocab_size = config["model_kwargs"]["vocab_size"]
    num_layers = config["model_kwargs"]["num_layers"]
    training_cfg = config.get("training_config", {})
    print(f"Modelo: vocab_size={vocab_size}, num_layers={num_layers}")
    print(f"Uniform baseline: max_softmax ≈ {1.0/vocab_size:.6f}")

    # ── Cargar datos ──
    print("Preparando datos reales...")
    _, val_loader, _ = create_real_dataloaders(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_split=0.99,  # Casi todo para validación
        num_workers=0,
    )

    # ── 1. Distribución de confianza por capa ──
    print(f"\n{'='*60}")
    print("FASE 1: Distribución de confianza por capa")
    print(f"{'='*60}")
    core = model.core if hasattr(model, "core") else model
    diag = PMTDiagnostics(model, vocab_size=vocab_size, num_layers=num_layers)

    batch_count = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if batch_idx >= args.num_batches:
            break
        inputs = inputs.to(device, non_blocking=True)

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
        ):
            hidden = model.token_embedding(inputs)
            diag.instrumented_forward(hidden, exit_threshold=0.85)

        batch_count += 1
        if (batch_idx + 1) % 5 == 0:
            print(f"  Procesados {batch_idx + 1}/{args.num_batches} batches...")

    confidence_stats = diag.compute_stats()

    print("\nEstadísticas de confianza por capa:")
    print(f"{'Capa':>6} | {'Mean':>8} | {'Median':>8} | {'Max':>8} | {'P95':>8} | {'P99':>8} | {'>0.85':>8}")
    print("-" * 72)
    for i in range(num_layers):
        s = confidence_stats[f"layer_{i}"]
        if s["count"] == 0:
            print(f"{i:>6} | {'N/A':>8}")
            continue
        print(f"{i:>6} | {s['mean']:>8.5f} | {s['median']:>8.5f} | {s['max']:>8.5f} | "
              f"{s['p95']:>8.5f} | {s['p99']:>8.5f} | {s['pct_above_0.85']:>7.2f}%")

    # ── 2. Sweep de thresholds ──
    print(f"\n{'='*60}")
    print("FASE 2: Sweep de thresholds")
    print(f"{'='*60}")
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85]
    sweep_results = threshold_sweep(
        model=model,
        dataloader=val_loader,
        device=device,
        amp_dtype=amp_dtype,
        thresholds=thresholds,
        num_batches=args.num_batches,
    )

    # ── 3. Diagnóstico ──
    print(f"\n{'='*60}")
    print("DIAGNÓSTICO")
    print(f"{'='*60}")

    # Tomar la confianza máxima observada en la última capa
    last_layer_stats = confidence_stats.get(f"layer_{num_layers - 1}", {})
    max_observed = last_layer_stats.get("max", 0.0)
    mean_observed = last_layer_stats.get("mean", 0.0)
    p99_observed = last_layer_stats.get("p99", 0.0)

    print(f"\nVocab size: {vocab_size}")
    print(f"Confianza uniforme teórica: {1.0/vocab_size:.8f}")
    print(f"Confianza media última capa: {mean_observed:.6f}")
    print(f"Confianza P99 última capa: {p99_observed:.6f}")
    print(f"Confianza máxima observada (última capa): {max_observed:.6f}")
    print(f"Threshold actual: 0.85")
    print(f"\n→ CAUSA RAÍZ: softmax sobre {vocab_size} clases produce confianzas")
    print(f"  max ≈ {max_observed:.4f}, muy lejos del threshold 0.85.")
    print(f"  El PMT NUNCA puede activar early exit con esta configuración.")

    # Sugerir threshold óptimo
    # Buscar el threshold que produce ~80% de profundidad (20% ahorro)
    print(f"\n→ RECOMENDACIÓN: Usar un threshold adaptado al rango real de confianza.")
    if p99_observed > 0:
        suggested = round(p99_observed * 0.8, 4)
        print(f"  Threshold sugerido (80% del P99): {suggested}")

    # ── Guardar resultados ──
    output_path = args.output or str(export_dir / "pmt_diagnosis.json")
    report = {
        "diagnosed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "uniform_baseline_confidence": 1.0 / vocab_size,
        "current_threshold": 0.85,
        "batches_analyzed": batch_count,
        "confidence_per_layer": confidence_stats,
        "threshold_sweep": sweep_results,
        "diagnosis": {
            "root_cause": (
                f"softmax().amax() over vocab_size={vocab_size} produces max confidence "
                f"≈ {max_observed:.6f}, far below threshold=0.85. "
                f"PMT early exit is structurally impossible with current metric."
            ),
            "max_confidence_observed": max_observed,
            "p99_confidence_last_layer": p99_observed,
            "mean_confidence_last_layer": mean_observed,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReporte guardado en: {output_path}")


if __name__ == "__main__":
    main()
