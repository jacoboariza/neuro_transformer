import tempfile
import unittest
from pathlib import Path

import torch

from experiments.train_real import (
    EXPORT_CONFIG_FILENAME,
    EXPORT_TOKENIZER_DIRNAME,
    EXPORT_WEIGHTS_FILENAME,
    NeuroModelV2ForLM,
    export_model_bundle,
    load_exported_model_bundle,
    load_safetensors_file,
    save_safetensors_file,
)


class _DummyTokenizer:
    def save_pretrained(self, save_directory: str) -> None:
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer_config.json").write_text('{"dummy": true}\n', encoding="utf-8")


@unittest.skipIf(
    save_safetensors_file is None or load_safetensors_file is None,
    "safetensors no disponible en el entorno",
)
class TrainRealExportTests(unittest.TestCase):
    def test_export_bundle_writes_expected_artifacts_and_load_roundtrip(self):
        torch.manual_seed(123)
        model = NeuroModelV2ForLM(vocab_size=32, embed_dim=16, num_layers=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            export_dir = tmp_path / "bundle"
            checkpoint_path = tmp_path / "best_model.pt"

            exported_path = export_model_bundle(
                export_dir=export_dir,
                model=model,
                tokenizer=_DummyTokenizer(),
                training_config={"dataset_name": "dummy_dataset", "tokenizer_name": "dummy_tokenizer"},
                checkpoint_path=checkpoint_path,
                epoch=3,
                train_loss=1.23,
                val_loss=1.11,
            )

            self.assertEqual(exported_path, export_dir)
            self.assertTrue((export_dir / EXPORT_WEIGHTS_FILENAME).exists())
            self.assertTrue((export_dir / EXPORT_CONFIG_FILENAME).exists())
            self.assertTrue((export_dir / EXPORT_TOKENIZER_DIRNAME / "tokenizer_config.json").exists())

            loaded_model, artifact_config = load_exported_model_bundle(export_dir=export_dir, device="cpu")
            self.assertEqual(artifact_config["architecture"], "NeuroModelV2ForLM")
            self.assertEqual(artifact_config["model_kwargs"]["vocab_size"], 32)
            self.assertEqual(artifact_config["checkpoint_file"], checkpoint_path.name)

            original_state_dict = model.state_dict()
            loaded_state_dict = loaded_model.state_dict()
            self.assertEqual(set(original_state_dict.keys()), set(loaded_state_dict.keys()))
            for key in original_state_dict:
                self.assertTrue(
                    torch.equal(original_state_dict[key].cpu(), loaded_state_dict[key].cpu()),
                    msg=f"Peso no coincide tras export/load para clave: {key}",
                )

    def test_load_bundle_fails_when_config_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                load_exported_model_bundle(export_dir=Path(tmp_dir), device="cpu")


if __name__ == "__main__":
    unittest.main()
