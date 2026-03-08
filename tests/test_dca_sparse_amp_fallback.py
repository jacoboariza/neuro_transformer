import unittest
from unittest.mock import patch

import torch

from models.dca import FixedSparseLinear


class DcaSparseAmpFallbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.layer = FixedSparseLinear(in_features=8, out_features=8, sparsity=0.5, bias=False)
        self.x2d = torch.randn(6, 8)
        self.sparse_weight = self.layer.sparse_weight()

    def test_sparse_mm_fallback_uses_fp32_after_amp_not_implemented_error(self):
        real_sparse_mm = torch.sparse.mm
        call_counter = {"count": 0}

        def _mock_sparse_mm(weight: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                raise NotImplementedError('"addmm_sparse_cuda" not implemented for \'BFloat16\'')
            return real_sparse_mm(weight, matrix)

        with patch("torch.sparse.mm", side_effect=_mock_sparse_mm):
            projected = self.layer._sparse_mm_with_fallback(sparse_weight=self.sparse_weight, x2d=self.x2d)

        self.assertEqual(call_counter["count"], 2)
        self.assertEqual(projected.dtype, torch.float32)
        self.assertEqual(projected.shape, (self.x2d.shape[0], self.layer.out_features))

    def test_sparse_mm_fallback_reraises_unrelated_not_implemented_error(self):
        with patch("torch.sparse.mm", side_effect=NotImplementedError("unrelated backend gap")):
            with self.assertRaises(NotImplementedError):
                self.layer._sparse_mm_with_fallback(sparse_weight=self.sparse_weight, x2d=self.x2d)


if __name__ == "__main__":
    unittest.main()
