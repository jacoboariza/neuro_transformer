import unittest

import torch

from models.dca import DCA_Layer
from neuro_architectures_v2 import NeuroModelV2, VLM_VicariousStudent
from utils.compliance import CANONICAL_DATASET, CANONICAL_TOKENIZER, build_compliance_report
from utils.profiler import DeviceTimer, estimate_flops_torch_profiler


class RequirementsComplianceTests(unittest.TestCase):
    def test_r2_dca_uses_sparse_kernel(self):
        model = DCA_Layer(embed_dim=16, sparsity=0.9)
        self.assertTrue(hasattr(model, "sparse_linear"))

        sparse_weight = model.sparse_linear.sparse_weight()
        self.assertTrue(sparse_weight.is_sparse)

        x = torch.randn(2, 8, 16)
        out = model(x)
        self.assertEqual(tuple(out.shape), tuple(x.shape))

    def test_r3_pmt_token_masking_depth_behavior(self):
        model = NeuroModelV2(embed_dim=16, num_classes=32, num_layers=4)
        x = torch.randn(2, 6, 16)

        logits_easy, layers_easy, _ = model(x, exit_threshold=0.0)
        self.assertEqual(tuple(logits_easy.shape), (2, 6, 32))
        self.assertAlmostEqual(layers_easy, 1.0, places=4)

        logits_hard, layers_hard, _ = model(x, exit_threshold=1.1)
        self.assertEqual(tuple(logits_hard.shape), (2, 6, 32))
        self.assertAlmostEqual(layers_hard, 4.0, places=4)

    def test_r4_vlm_uses_detach_for_gradient_isolation(self):
        student = VLM_VicariousStudent(embed_dim=8, student_dim=4)
        hidden = torch.randn(2, 5, 8, requires_grad=True)

        loss = student.observe_and_learn(hidden)
        loss.backward()

        self.assertIsNone(hidden.grad)

        has_student_grad = any(
            parameter.grad is not None and torch.any(parameter.grad != 0)
            for parameter in student.parameters()
        )
        self.assertTrue(has_student_grad)

    def test_r5_timer_and_profiler_cpu_backend(self):
        device = torch.device("cpu")
        timer = DeviceTimer(device)

        marker = timer.start()
        _ = torch.randn(128, 128) @ torch.randn(128, 128)
        elapsed_ms = timer.stop(marker)
        self.assertGreater(elapsed_ms, 0.0)

        linear = torch.nn.Linear(16, 16)
        sample = torch.randn(4, 16)
        flops = estimate_flops_torch_profiler(lambda: linear(sample), device=device)
        self.assertGreaterEqual(flops, 0.0)

    def test_compliance_report_for_neuromodelv2(self):
        model = NeuroModelV2(embed_dim=16, num_classes=32, num_layers=2)
        report = build_compliance_report(
            model_name="NeuroModelV2",
            model=model,
            dataset_name=CANONICAL_DATASET,
            tokenizer_name=CANONICAL_TOKENIZER,
            device=torch.device("cpu"),
            profiling_backend="perf_counter",
        )

        as_dict = report.to_dict()
        self.assertTrue(as_dict["R1_RealData"])
        self.assertTrue(as_dict["R2_SparseDCA"])
        self.assertTrue(as_dict["R3_TokenMaskingPMT"])
        self.assertTrue(as_dict["R4_VLMDetach"])
        self.assertTrue(as_dict["R5_PreciseProfiling"])
        self.assertTrue(as_dict["EligibleForRanking"])

    def test_r1_non_canonical_dataset_is_not_eligible(self):
        model = NeuroModelV2(embed_dim=16, num_classes=32, num_layers=2)
        report = build_compliance_report(
            model_name="NeuroModelV2",
            model=model,
            dataset_name="wikimedia/wikipedia",
            device=torch.device("cpu"),
            profiling_backend="perf_counter",
        )

        as_dict = report.to_dict()
        self.assertFalse(as_dict["R1_RealData"])
        self.assertFalse(as_dict["EligibleForRanking"])

    def test_r1_non_canonical_tokenizer_is_not_eligible(self):
        model = NeuroModelV2(embed_dim=16, num_classes=32, num_layers=2)
        report = build_compliance_report(
            model_name="NeuroModelV2",
            model=model,
            dataset_name=CANONICAL_DATASET,
            tokenizer_name="gpt2",
            device=torch.device("cpu"),
            profiling_backend="perf_counter",
        )

        as_dict = report.to_dict()
        self.assertFalse(as_dict["R1_RealData"])
        self.assertFalse(as_dict["EligibleForRanking"])


if __name__ == "__main__":
    unittest.main()
