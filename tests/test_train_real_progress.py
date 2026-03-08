import unittest

from experiments.train_real import format_train_progress


class TrainRealProgressTests(unittest.TestCase):
    def test_format_train_progress_reports_percentage_and_eta(self):
        line = format_train_progress(
            epoch=2,
            total_epochs=5,
            batch_idx=50,
            total_batches=100,
            elapsed_seconds=20.0,
            running_train_loss=1.2345,
        )

        self.assertIn("[epoch 002/005]", line)
        self.assertIn("avance= 50.0%", line)
        self.assertIn("(50/100)", line)
        self.assertIn("loss=1.2345", line)
        self.assertIn("elapsed=20.0s", line)
        self.assertIn("eta=20.0s", line)

    def test_format_train_progress_clamps_completion_to_100_percent(self):
        line = format_train_progress(
            epoch=1,
            total_epochs=1,
            batch_idx=200,
            total_batches=100,
            elapsed_seconds=33.0,
            running_train_loss=0.9,
        )

        self.assertIn("avance=100.0%", line)
        self.assertIn("eta=0.0s", line)


if __name__ == "__main__":
    unittest.main()
