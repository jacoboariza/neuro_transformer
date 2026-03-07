from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd


SIZE_ORDER = {
    "Micro (6M)": 0,
    "Mini (15M)": 1,
    "Small (35M)": 2,
    "Base (85M)": 3,
    "Smol (135M)": 4,
    "Reference (135M)": 5,
}

BIO_INSPIRED_ARCHITECTURES = {"DCA", "MOPN", "SCT", "GMA_MoE"}


def _architecture_from_model(model_name: str, size_category: str) -> str:
    suffix = size_category.replace(" ", "")
    expected_tail = f"_{suffix}"
    if suffix and model_name.endswith(expected_tail):
        return model_name[: -len(expected_tail)]
    return model_name


def _size_rank(size_category: str) -> Tuple[int, float]:
    if size_category in SIZE_ORDER:
        return (0, float(SIZE_ORDER[size_category]))

    match = re.search(r"\((\d+(?:\.\d+)?)M\)", size_category)
    if match is not None:
        return (1, float(match.group(1)))

    return (2, float("inf"))


def build_crossover_table(
    results_df: pd.DataFrame,
    target_loss: float = 4.08,
    loss_column: str = "FinalLoss",
    include_non_bio_models: bool = False,
    architectures: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    required_columns = {"Model", "SizeCategory", loss_column}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for crossover analysis: {sorted(missing_columns)}")

    frame = results_df.copy()
    frame[loss_column] = pd.to_numeric(frame[loss_column], errors="coerce")
    frame = frame.dropna(subset=[loss_column])

    if "Status" in frame.columns:
        status_ok = frame["Status"].fillna("ok").astype(str).str.startswith("ok")
        frame = frame[status_ok]

    frame["Architecture"] = frame.apply(
        lambda row: _architecture_from_model(
            model_name=str(row.get("Model", "")),
            size_category=str(row.get("SizeCategory", "")),
        ),
        axis=1,
    )

    if architectures is not None:
        allowed_architectures = {arch.strip() for arch in architectures if arch.strip()}
        frame = frame[frame["Architecture"].isin(allowed_architectures)]
    elif not include_non_bio_models:
        frame = frame[frame["Architecture"].isin(BIO_INSPIRED_ARCHITECTURES)]

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Architecture",
                "TargetLoss",
                "CrossoverFound",
                "CrossoverModel",
                "CrossoverSizeCategory",
                "CrossoverLoss",
                "BestObservedModel",
                "BestObservedSizeCategory",
                "BestObservedLoss",
                "LossGapToTarget",
                "TrainableParams",
                "MicroBatchSize",
                "GradAccumSteps",
                "EffectiveBatchSize",
            ]
        )

    frame["__size_rank"] = frame["SizeCategory"].astype(str).map(_size_rank)
    frame = frame.sort_values(by=["Architecture", "__size_rank"], kind="stable")

    records = []
    for architecture, group in frame.groupby("Architecture", sort=False):
        ordered = group.sort_values(by=["__size_rank"], kind="stable")
        best_row = ordered.loc[ordered[loss_column].idxmin()]
        crossed = ordered[ordered[loss_column] <= target_loss]

        crossover_row = crossed.iloc[0] if not crossed.empty else None

        observed_loss = float(best_row[loss_column])
        record = {
            "Architecture": architecture,
            "TargetLoss": float(target_loss),
            "CrossoverFound": crossover_row is not None,
            "CrossoverModel": crossover_row["Model"] if crossover_row is not None else pd.NA,
            "CrossoverSizeCategory": crossover_row["SizeCategory"] if crossover_row is not None else pd.NA,
            "CrossoverLoss": float(crossover_row[loss_column]) if crossover_row is not None else pd.NA,
            "BestObservedModel": best_row["Model"],
            "BestObservedSizeCategory": best_row["SizeCategory"],
            "BestObservedLoss": observed_loss,
            "LossGapToTarget": observed_loss - float(target_loss),
            "TrainableParams": crossover_row["TrainableParams"] if crossover_row is not None and "TrainableParams" in ordered.columns else pd.NA,
            "MicroBatchSize": crossover_row["MicroBatchSize"] if crossover_row is not None and "MicroBatchSize" in ordered.columns else pd.NA,
            "GradAccumSteps": crossover_row["GradAccumSteps"] if crossover_row is not None and "GradAccumSteps" in ordered.columns else pd.NA,
            "EffectiveBatchSize": crossover_row["EffectiveBatchSize"] if crossover_row is not None and "EffectiveBatchSize" in ordered.columns else pd.NA,
            "__size_rank": crossover_row["__size_rank"] if crossover_row is not None else (99, float("inf")),
        }
        records.append(record)

    output = pd.DataFrame(records)
    output = output.sort_values(by=["CrossoverFound", "__size_rank", "BestObservedLoss"], ascending=[False, True, True], kind="stable")
    return output.drop(columns=["__size_rank"]).reset_index(drop=True)


def analyze_crossover(
    input_csv: str,
    target_loss: float = 4.08,
    loss_column: str = "FinalLoss",
    include_non_bio_models: bool = False,
    architectures: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    results_df = pd.read_csv(input_csv)
    return build_crossover_table(
        results_df=results_df,
        target_loss=target_loss,
        loss_column=loss_column,
        include_non_bio_models=include_non_bio_models,
        architectures=architectures,
    )


def _parse_architectures(raw_value: Optional[str]) -> Optional[Sequence[str]]:
    if raw_value is None:
        return None
    tokens = [token.strip() for token in raw_value.split(",")]
    return [token for token in tokens if token]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scaling-law crossover point from benchmark CSV.")
    parser.add_argument("--input-csv", type=str, default="benchmark_results.csv", help="Path to benchmark results CSV")
    parser.add_argument("--output-csv", type=str, default="", help="Path to save crossover summary CSV")
    parser.add_argument("--target-loss", type=float, default=4.08, help="Target loss threshold")
    parser.add_argument("--loss-column", type=str, default="FinalLoss", help="Loss column to evaluate")
    parser.add_argument("--include-non-bio", action="store_true", help="Include non bio-inspired models")
    parser.add_argument("--architectures", type=str, default=None, help="Comma-separated architecture filter")
    args = parser.parse_args()

    selected_architectures = _parse_architectures(args.architectures)
    summary_df = analyze_crossover(
        input_csv=args.input_csv,
        target_loss=args.target_loss,
        loss_column=args.loss_column,
        include_non_bio_models=args.include_non_bio,
        architectures=selected_architectures,
    )

    if summary_df.empty:
        print("No rows available for crossover analysis.")
    else:
        print("Crossover summary:")
        print(summary_df.to_string(index=False))

    output_path = Path(args.output_csv) if args.output_csv else Path(args.input_csv).with_name(f"{Path(args.input_csv).stem}_crossover.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"Crossover CSV exported to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
