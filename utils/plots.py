from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_benchmark_results(
    csv_path: str = "benchmark_results.csv",
    output_dir: str = ".",
) -> dict:
    data = pd.read_csv(csv_path)

    required_columns = {"Model", "TrainableParams", "TrainTimeSeconds"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columnas faltantes en CSV: {sorted(missing_columns)}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_plot_path = out_dir / "benchmark_params.png"
    time_plot_path = out_dir / "benchmark_time.png"
    flops_plot_path = out_dir / "benchmark_flops.png"
    tokens_per_sec_plot_path = out_dir / "benchmark_tokens_per_sec.png"
    real_case_acc_plot_path = out_dir / "benchmark_real_case_accuracy.png"
    real_case_loss_plot_path = out_dir / "benchmark_real_case_loss.png"
    compliance_plot_path = out_dir / "benchmark_compliance.png"

    plt.figure(figsize=(10, 5))
    plt.bar(data["Model"], data["TrainableParams"])
    plt.title("Comparación de parámetros entrenables por modelo")
    plt.xlabel("Modelo")
    plt.ylabel("Parámetros entrenables")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(params_plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(data["Model"], data["TrainTimeSeconds"])
    plt.title("Comparación de tiempo total de entrenamiento")
    plt.xlabel("Modelo")
    plt.ylabel("Tiempo (segundos)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(time_plot_path, dpi=150)
    plt.close()

    plots = {
        "params_plot": str(params_plot_path),
        "time_plot": str(time_plot_path),
    }

    if "TrainFLOPs" in data.columns:
        flops_data = data.dropna(subset=["TrainFLOPs"])
        if not flops_data.empty:
            plt.figure(figsize=(10, 5))
            plt.bar(flops_data["Model"], flops_data["TrainFLOPs"])
            plt.title("Comparación de FLOPs de entrenamiento")
            plt.xlabel("Modelo")
            plt.ylabel("FLOPs")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(flops_plot_path, dpi=150)
            plt.close()
            print(f"Grafico de FLOPs guardado en: {flops_plot_path.resolve()}")
            plots["flops_plot"] = str(flops_plot_path)

    if "TrainTokensPerSecond" in data.columns:
        plt.figure(figsize=(10, 5))
        plt.bar(data["Model"], data["TrainTokensPerSecond"])
        plt.title("Comparación de throughput de entrenamiento")
        plt.xlabel("Modelo")
        plt.ylabel("Tokens por segundo")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(tokens_per_sec_plot_path, dpi=150)
        plt.close()
        print(f"Grafico de throughput guardado en: {tokens_per_sec_plot_path.resolve()}")
        plots["tokens_per_sec_plot"] = str(tokens_per_sec_plot_path)

    if "RealCaseAccuracy" in data.columns:
        real_acc = data.dropna(subset=["RealCaseAccuracy"])
        if not real_acc.empty:
            plt.figure(figsize=(10, 5))
            plt.bar(real_acc["Model"], real_acc["RealCaseAccuracy"])
            plt.title("Comparación de accuracy en casos reales")
            plt.xlabel("Modelo")
            plt.ylabel("Accuracy")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(real_case_acc_plot_path, dpi=150)
            plt.close()
            print(f"Grafico de casos reales (accuracy) guardado en: {real_case_acc_plot_path.resolve()}")
            plots["real_case_accuracy_plot"] = str(real_case_acc_plot_path)

    if "RealCaseLoss" in data.columns:
        real_loss = data.dropna(subset=["RealCaseLoss"])
        if not real_loss.empty:
            plt.figure(figsize=(10, 5))
            plt.bar(real_loss["Model"], real_loss["RealCaseLoss"])
            plt.title("Comparación de loss en casos reales")
            plt.xlabel("Modelo")
            plt.ylabel("Loss")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(real_case_loss_plot_path, dpi=150)
            plt.close()
            print(f"Grafico de casos reales (loss) guardado en: {real_case_loss_plot_path.resolve()}")
            plots["real_case_loss_plot"] = str(real_case_loss_plot_path)

    if "EligibleForRanking" in data.columns:
        compliance_flags = data["EligibleForRanking"].fillna(False).astype(bool)
        compliance_numeric = compliance_flags.astype(int)

        plt.figure(figsize=(10, 5))
        plt.bar(
            data["Model"],
            compliance_numeric,
            color=["#2e7d32" if flag else "#c62828" for flag in compliance_flags],
        )
        plt.title("Elegibilidad de compliance R1-R5")
        plt.xlabel("Modelo")
        plt.ylabel("EligibleForRanking (1=si, 0=no)")
        plt.yticks([0, 1])
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(compliance_plot_path, dpi=150)
        plt.close()
        print(f"Grafico de compliance guardado en: {compliance_plot_path.resolve()}")
        plots["compliance_plot"] = str(compliance_plot_path)

    print(f"Grafico de parametros guardado en: {params_plot_path.resolve()}")
    print(f"Grafico de tiempo guardado en: {time_plot_path.resolve()}")
    return plots


if __name__ == "__main__":
    plot_benchmark_results()
