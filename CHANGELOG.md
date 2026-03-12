# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- _Sin cambios registrados todavía._

## [0.5.0] - 2026-03-12

### Added
- **`entropy_confidence()` function** in `neuro_architectures_v2.py`:
  - Normalizes confidence via entropy: `1 - H(p)/log(V)`, producing [0,1] values independent of vocabulary size.
  - Replaces `max(softmax)` which was unusable for large vocabularies (49K+ tokens).
- **New evaluation script** `experiments/eval_dca_real.py`:
  - Evaluates exported DCA model bundles on real FineWeb-Edu data.
  - CUDA runtime probe with robust fallback, AMP autodetection, JSON metrics report.
  - CLI options for threshold, samples, device, AMP dtype.
- **New diagnostic script** `experiments/diagnose_pmt.py`:
  - Instruments PMT forward pass to capture confidence distribution per layer.
  - Threshold sweep with compute savings analysis.
  - JSON report with statistics and recommendations.
- **New test** `test_r3_entropy_confidence_is_vocab_size_independent` in `tests/test_r2_r5_compliance.py`:
  - Validates entropy confidence produces [0,1] for any vocab size.
  - Checks uniform logits → 0, one-hot logits → ~1.
- **Consolidated review** `docs/CONSOLIDATED_REVIEW.md` documenting architecture analysis, findings and action items.

### Changed
- `neuro_architectures_v2.py`: `PMT_EarlyExit.forward()` and `NeuroModelV2.forward()` now use `entropy_confidence` instead of `max(softmax)`.
- `experiments/train_real.py`: `compute_composite_loss` confidence reward uses `entropy_confidence`.
- `experiments/train_real.py`: default `--num-samples` increased from 50,000 to 200,000 for better model convergence.
- `.gitignore`: added patterns for generated eval/diagnosis JSON reports.
- `README.md`: documented `eval_dca_real.py` usage and capabilities.

### Fixed
- **Critical: PMT early exit never activated** — `max(softmax)` over 49K vocab produced mean confidence ~0.17, making `threshold=0.85` unreachable. With `entropy_confidence`, the metric is now meaningful and vocab-size independent. Backward compatible: existing models at `threshold=0.85` produce identical results (no early exit), but retraining with the new metric enables proper early exit learning.

### Security / Compliance
- No hardcoded credentials detected in tracked project files.
- Compliance R1-R5 remains fully satisfied.

### Notes
- No breaking API changes; existing trained models load and evaluate identically.
- Models trained with the new `entropy_confidence` metric will benefit from functional early exit at `threshold=0.85`.
- Retraining recommended to take advantage of the fix.

## [0.4.0] - 2026-03-08

### Added
- **Reusable model export bundle** in `experiments/train_real.py`:
  - `model.safetensors` export for model weights
  - `config.json` with model hyperparameters, best metrics, and training metadata
  - `tokenizer/` artifact export via `save_pretrained(...)`
  - `load_exported_model_bundle(...)` helper to reload exported models for downstream projects
  - New CLI option `--export-dir` (default: `<output-dir>/best_model_export`)
- **New tests** in `tests/test_train_real_export.py`:
  - Export+load roundtrip validation
  - Error handling when export config is missing
- **Training progress display** in `experiments/train_real.py`:
  - Progress lines per epoch with percentage, batch counters, running loss, elapsed time and ETA
- **New tests**:
  - `tests/test_train_real_progress.py` for progress formatting behavior
  - `tests/test_dca_sparse_amp_fallback.py` for sparse AMP fallback behavior

### Changed
- `experiments/train_real.py` now exports a reusable bundle each time a new best validation checkpoint is saved.
- `README.md` now documents exported artifact structure, downstream loading example and train progress output.

### Fixed
- `models/dca.py`: added fallback for CUDA sparse matmul under AMP when BF16/FP16 sparse kernels are unavailable (`addmm_sparse_cuda`).
- `neuro_architectures_v2.py`: fixed dtype mismatch in token-masking assignments during autocast.

### Security / Compliance
- No hardcoded credentials detected in tracked project files.

## [0.3.0] - 2026-03-04

### Added
- **Scaling Laws Benchmark** in `experiments/run_benchmark.py`:
  - Nested loops over model sizes (Micro 6M → Smol 135M) and architectures
  - Dynamic micro-batch sizing with OOM prevention for RTX 5070 Ti
  - Gradient accumulation support with AdamW optimizer
  - SizeCategory column and enhanced model naming (e.g., "SCT_Mini(15M)")
  - StackedArchitectureModel wrapper for multi-layer scaling
- **Crossover Analysis Utility** in `utils/crossover_analysis.py`:
  - Detects first size achieving target loss (default 4.08)
  - Filters bio-inspired architectures by default
  - CSV export with crossover points and loss gaps
  - CLI interface for custom analysis
- **Comprehensive Unit Tests**:
  - `tests/test_scaling_benchmark.py`: batch planning, model stacking, gradient accumulation
  - `tests/test_crossover_analysis.py`: crossover detection and CSV parsing
- **Updated Documentation**:
  - Scaling laws benchmark flow and OOM strategy in README
  - Crossover analysis usage examples
  - Enhanced benchmark CSV columns documentation
- **FlyWire Connectome Utility** in `utils/brain_downloader.py`:
  - Example script to query FlyWire synapses (`synapses_nt_v120`)
  - Builds a directed weighted neural graph and exports adjacency matrix
  - Notes authentication requirement through CAVEclient token flow

### Changed
- **Training Loop** (`experiments/train_loop.py`):
  - Switched to AdamW optimizer
  - Added configurable gradient accumulation
  - Fixed embed_dim inference for stacked models
  - SCT sleep_cycle logging only when executed
- **Benchmark Results**:
  - New columns: SizeCategory, MicroBatchSize, GradAccumSteps, EffectiveBatchSize
  - Enhanced model naming with size information
  - Improved OOM error handling and retry logging
- **Model Factories**:
  - Refactored to support dynamic parameter injection (embed_dim, num_layers, num_heads)
  - Consistent stacking behavior across all architectures
- **NeuroModelV2 CEN stability** (`neuro_architectures_v2.py`):
  - Fixed counterfactual branch selection to avoid `NoneType` failures when coherence score comparisons produce invalid values

### Security / Compliance
- No hardcoded credentials introduced.
- HF_TOKEN remains optional and externally provided.

### Notes
- No breaking API changes; this is a **minor** feature release.
- Backward compatibility preserved for existing benchmark scripts.

## [0.2.0] - 2026-02-28

### Added
- Real-data pipeline in `data/real_data.py` using Hugging Face `datasets` + `transformers` with fallback dataset support.
- Reusable model building blocks in `models/blocks.py`:
  - `RMSNorm`
  - `RotaryEmbedding`
  - `SwiGLUFeedForward`
  - `MultiHeadSelfAttentionRoPE`
- New multicapa wrapper `models/neuro_model.py` for stacking architectures and projecting to vocabulary.
- Compliance and profiling utilities:
  - `utils/compliance.py` with R1-R5 checks and ranking eligibility gate
  - `utils/profiler.py` with device-aware timing and FLOPs estimation via `torch.profiler`
- New high-performance training entrypoint `experiments/train_real.py` with:
  - mixed precision (`autocast` + optional `GradScaler`)
  - AdamW + cosine warmup scheduler
  - best-checkpoint saving
- Integration with `NeuroModelV2` (`neuro_architectures_v2.py`) in `train_real.py`.
- Composite loss in `train_real.py` that combines:
  - next-token prediction loss
  - vicarious loss regularization
  - PMT early-exit reward to encourage lower compute depth.

### Changed
- Modernized baseline and bio-inspired layers to use RoPE/RMSNorm/SwiGLU patterns:
  - `models/base_transformer.py`
  - `models/dca.py`
  - `models/mopn.py`
  - `models/sct.py`
  - `models/gma_moe.py`
- Updated benchmark outputs and plotting to include richer metrics and ranking.
- Benchmark CSV/report now include provenance columns for dataset/tokenizer (`Requested*`, `Resolved*`, `UsedFallbackDataset`) and compliance columns (`R1..R5`, `EligibleForRanking`).
- Composite ranking now enforces compliance gate: non-eligible runs are excluded from official ranking (`CompositeScore`/`CompositeRank` vacios).
- Updated `README.md` usage and architecture sections for NeuroModelV2 and PMT-aware training.
- Added release hygiene files:
  - `VERSION` (current release: `0.2.0`)
  - `.gitignore`

### Security / Compliance
- No hardcoded credentials introduced.
- `HF_TOKEN` remains optional and externally provided via environment.

### Notes
- No breaking API intended for current scripts; this is a **minor** feature release.
