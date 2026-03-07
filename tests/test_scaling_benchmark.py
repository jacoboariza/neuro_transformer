import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_benchmark import (
    StackedArchitectureModel,
    build_micro_batch_candidates,
    build_models,
    resolve_scaling_training_plan,
)
from experiments.train_loop import train_model


class TinyBlock(torch.nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ScalingBenchmarkTests(unittest.TestCase):
    def test_resolve_scaling_training_plan_uses_expected_cuda_caps(self):
        micro_batch, grad_accum = resolve_scaling_training_plan(
            size_name="Base (85M)",
            requested_batch_size=24,
            device_type="cuda",
        )
        self.assertEqual(micro_batch, 4)
        self.assertEqual(grad_accum, 6)

        micro_batch, grad_accum = resolve_scaling_training_plan(
            size_name="Smol (135M)",
            requested_batch_size=24,
            device_type="cuda",
        )
        self.assertEqual(micro_batch, 2)
        self.assertEqual(grad_accum, 12)

    def test_build_micro_batch_candidates_halves_until_one(self):
        self.assertEqual(build_micro_batch_candidates(8), [8, 4, 2, 1])
        self.assertEqual(build_micro_batch_candidates(1), [1])

    def test_build_models_creates_stacked_models_with_expected_shapes(self):
        factories = build_models(embed_dim=64, num_layers=3, num_heads=8)
        self.assertEqual(set(factories.keys()), {"Transformer", "DCA", "MOPN", "SCT", "GMA_MoE"})

        for model_name, factory in factories.items():
            model = factory()
            self.assertIsInstance(model, StackedArchitectureModel)
            self.assertEqual(len(model.layers), 3)

            x = torch.randn(2, 5, 64)
            y = model(x)
            self.assertEqual(tuple(y.shape), tuple(x.shape), msg=f"Shape mismatch in {model_name}")

    def test_sleep_cycle_only_reports_true_when_layer_supports_it(self):
        no_sleep_model = StackedArchitectureModel(
            embed_dim=16,
            num_layers=2,
            layer_builder=lambda: TinyBlock(embed_dim=16),
        )
        self.assertFalse(no_sleep_model.sleep_cycle())

        sct_model = build_models(embed_dim=32, num_layers=2, num_heads=8)["SCT"]()
        self.assertTrue(sct_model.sleep_cycle())

    def test_train_model_supports_gradient_accumulation(self):
        inputs = torch.randint(0, 32, (32, 10), dtype=torch.long)
        targets = torch.randint(0, 32, (32, 10), dtype=torch.long)
        dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)

        model = TinyBlock(embed_dim=16)
        history = train_model(
            model=model,
            dataloader=dataloader,
            epochs=1,
            lr=1e-3,
            device="cpu",
            grad_accum_steps=2,
        )

        self.assertEqual(len(history["loss"]), 1)
        self.assertEqual(len(history["epoch_tokens"]), 1)
        self.assertGreater(history["epoch_tokens"][0], 0)

    def test_train_model_with_stacked_mopn_uses_wrapper_embed_dim(self):
        inputs = torch.randint(0, 48, (16, 6), dtype=torch.long)
        targets = torch.randint(0, 48, (16, 6), dtype=torch.long)
        dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)

        mopn_model = build_models(embed_dim=32, num_layers=1, num_heads=8)["MOPN"]()
        history = train_model(
            model=mopn_model,
            dataloader=dataloader,
            epochs=1,
            lr=1e-3,
            device="cpu",
            grad_accum_steps=1,
        )

        self.assertEqual(len(history["loss"]), 1)
        self.assertGreater(history["epoch_tokens"][0], 0)


if __name__ == "__main__":
    unittest.main()
