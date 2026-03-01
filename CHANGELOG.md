# Changelog

All notable changes to this project will be documented in this file.

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
