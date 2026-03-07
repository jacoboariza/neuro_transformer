import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.crossover_analysis import analyze_crossover, build_crossover_table


class CrossoverAnalysisTests(unittest.TestCase):
    def test_build_crossover_table_detects_first_crossing_size(self):
        frame = pd.DataFrame(
            [
                {"Model": "DCA_Micro(6M)", "SizeCategory": "Micro (6M)", "FinalLoss": 5.10, "Status": "ok", "TrainableParams": 1_000_000},
                {"Model": "DCA_Mini(15M)", "SizeCategory": "Mini (15M)", "FinalLoss": 4.30, "Status": "ok", "TrainableParams": 2_000_000},
                {"Model": "DCA_Small(35M)", "SizeCategory": "Small (35M)", "FinalLoss": 4.01, "Status": "ok", "TrainableParams": 3_000_000},
            ]
        )

        summary = build_crossover_table(results_df=frame, target_loss=4.08)
        self.assertEqual(len(summary), 1)

        row = summary.iloc[0]
        self.assertTrue(row["CrossoverFound"])
        self.assertEqual(row["CrossoverSizeCategory"], "Small (35M)")
        self.assertEqual(row["CrossoverModel"], "DCA_Small(35M)")

    def test_build_crossover_table_ignores_non_bio_models_by_default(self):
        frame = pd.DataFrame(
            [
                {"Model": "Transformer_Micro(6M)", "SizeCategory": "Micro (6M)", "FinalLoss": 3.90, "Status": "ok"},
                {"Model": "SCT_Micro(6M)", "SizeCategory": "Micro (6M)", "FinalLoss": 4.20, "Status": "ok"},
                {"Model": "SCT_Small(35M)", "SizeCategory": "Small (35M)", "FinalLoss": 4.05, "Status": "ok"},
            ]
        )

        summary = build_crossover_table(results_df=frame, target_loss=4.08)
        self.assertEqual(summary["Architecture"].tolist(), ["SCT"])

        summary_with_non_bio = build_crossover_table(
            results_df=frame,
            target_loss=4.08,
            include_non_bio_models=True,
        )
        self.assertEqual(set(summary_with_non_bio["Architecture"].tolist()), {"SCT", "Transformer"})

    def test_build_crossover_table_reports_gap_when_not_crossed(self):
        frame = pd.DataFrame(
            [
                {"Model": "GMA_MoE_Micro(6M)", "SizeCategory": "Micro (6M)", "FinalLoss": 4.90, "Status": "ok"},
                {"Model": "GMA_MoE_Smol(135M)", "SizeCategory": "Smol (135M)", "FinalLoss": 4.20, "Status": "ok"},
            ]
        )

        summary = build_crossover_table(results_df=frame, target_loss=4.08)
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]

        self.assertFalse(row["CrossoverFound"])
        self.assertTrue(pd.isna(row["CrossoverSizeCategory"]))
        self.assertAlmostEqual(row["BestObservedLoss"], 4.20, places=6)
        self.assertAlmostEqual(row["LossGapToTarget"], 0.12, places=6)

    def test_analyze_crossover_loads_csv_and_returns_summary(self):
        frame = pd.DataFrame(
            [
                {"Model": "MOPN_Micro(6M)", "SizeCategory": "Micro (6M)", "FinalLoss": 4.50, "Status": "ok"},
                {"Model": "MOPN_Base(85M)", "SizeCategory": "Base (85M)", "FinalLoss": 4.00, "Status": "ok"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "benchmark_results.csv"
            frame.to_csv(csv_path, index=False)
            summary = analyze_crossover(input_csv=str(csv_path), target_loss=4.08)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["CrossoverSizeCategory"], "Base (85M)")


if __name__ == "__main__":
    unittest.main()
