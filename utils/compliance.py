from dataclasses import dataclass
from typing import Dict, Optional

import torch


CANONICAL_DATASET = "HuggingFaceFW/fineweb-edu"
CANONICAL_TOKENIZER = "HuggingFaceTB/SmolLM-135M"


@dataclass
class ComplianceReport:
    r1_real_data_pipeline: bool
    r2_sparse_dca: bool
    r3_token_masking_pmt: bool
    r4_vlm_detach: bool
    r5_precise_profiling: bool

    @property
    def eligible_for_ranking(self) -> bool:
        return (
            self.r1_real_data_pipeline
            and self.r2_sparse_dca
            and self.r3_token_masking_pmt
            and self.r4_vlm_detach
            and self.r5_precise_profiling
        )

    def to_dict(self) -> Dict[str, bool]:
        return {
            "R1_RealData": self.r1_real_data_pipeline,
            "R2_SparseDCA": self.r2_sparse_dca,
            "R3_TokenMaskingPMT": self.r3_token_masking_pmt,
            "R4_VLMDetach": self.r4_vlm_detach,
            "R5_PreciseProfiling": self.r5_precise_profiling,
            "EligibleForRanking": self.eligible_for_ranking,
        }


def is_r1_real_data(dataset_name: str, tokenizer_name: Optional[str] = None) -> bool:
    if dataset_name != CANONICAL_DATASET:
        return False
    if tokenizer_name is not None and tokenizer_name != CANONICAL_TOKENIZER:
        return False
    return True


def is_r2_sparse_dca_model(model_name: str, model: torch.nn.Module) -> bool:
    if model_name.lower() != "dca":
        return True
    return bool(hasattr(model, "sparse_linear") or hasattr(model, "uses_sparse_dca"))


def is_r3_token_masking_model(model: torch.nn.Module) -> bool:
    if hasattr(model, "uses_token_masking"):
        return bool(getattr(model, "uses_token_masking"))
    return True


def is_r4_vlm_detach_model(model: torch.nn.Module) -> bool:
    if hasattr(model, "uses_vlm_detach"):
        return bool(getattr(model, "uses_vlm_detach"))
    return True


def is_r5_precise_profiling(device: torch.device, profiling_backend: str) -> bool:
    if device.type != "cuda":
        return True
    return profiling_backend == "cuda_event"


def build_compliance_report(
    *,
    model_name: str,
    model: torch.nn.Module,
    dataset_name: str,
    tokenizer_name: Optional[str] = None,
    device: torch.device,
    profiling_backend: str,
) -> ComplianceReport:
    return ComplianceReport(
        r1_real_data_pipeline=is_r1_real_data(dataset_name=dataset_name, tokenizer_name=tokenizer_name),
        r2_sparse_dca=is_r2_sparse_dca_model(model_name=model_name, model=model),
        r3_token_masking_pmt=is_r3_token_masking_model(model=model),
        r4_vlm_detach=is_r4_vlm_detach_model(model=model),
        r5_precise_profiling=is_r5_precise_profiling(device=device, profiling_backend=profiling_backend),
    )
