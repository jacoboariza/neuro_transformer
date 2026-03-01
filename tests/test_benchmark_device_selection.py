import unittest
from unittest.mock import patch

from experiments.run_benchmark import select_benchmark_device


class BenchmarkDeviceSelectionTests(unittest.TestCase):
    @patch("experiments.run_benchmark.torch.cuda.is_available", return_value=False)
    def test_falls_back_to_cpu_when_cuda_not_available(self, _mock_is_available):
        device, reason = select_benchmark_device()

        self.assertEqual(device, "cpu")
        self.assertIn("CUDA no está disponible", reason)

    @patch("experiments.run_benchmark.torch.cuda.get_arch_list", return_value=["sm_80", "sm_90"])
    @patch("experiments.run_benchmark.torch.cuda.get_device_capability", return_value=(12, 0))
    @patch("experiments.run_benchmark.torch.cuda.is_available", return_value=True)
    def test_falls_back_to_cpu_when_cuda_arch_not_supported(
        self,
        _mock_is_available,
        _mock_get_capability,
        _mock_get_arch_list,
    ):
        device, reason = select_benchmark_device()

        self.assertEqual(device, "cpu")
        self.assertIn("sm_120", reason)

    @patch("experiments.run_benchmark._probe_cuda_runtime", side_effect=RuntimeError("no kernel image"))
    @patch("experiments.run_benchmark.torch.cuda.get_arch_list", return_value=["sm_90", "sm_120"])
    @patch("experiments.run_benchmark.torch.cuda.get_device_capability", return_value=(12, 0))
    @patch("experiments.run_benchmark.torch.cuda.is_available", return_value=True)
    def test_falls_back_to_cpu_when_cuda_probe_fails(
        self,
        _mock_is_available,
        _mock_get_capability,
        _mock_get_arch_list,
        _mock_probe,
    ):
        device, reason = select_benchmark_device()

        self.assertEqual(device, "cpu")
        self.assertIn("no utilizable", reason)

    @patch("experiments.run_benchmark._probe_cuda_runtime")
    @patch("experiments.run_benchmark.torch.cuda.get_arch_list", return_value=["sm_90", "sm_120"])
    @patch("experiments.run_benchmark.torch.cuda.get_device_capability", return_value=(12, 0))
    @patch("experiments.run_benchmark.torch.cuda.is_available", return_value=True)
    def test_uses_cuda_when_arch_supported_and_probe_passes(
        self,
        _mock_is_available,
        _mock_get_capability,
        _mock_get_arch_list,
        mock_probe,
    ):
        device, reason = select_benchmark_device()

        self.assertEqual(device, "cuda")
        self.assertIsNone(reason)
        mock_probe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
