from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticNextTokenDataset(Dataset):
    """
    Dataset sintético para next-token prediction con patrones controlados.

    Cada muestra se genera como una secuencia de longitud (seq_len + 1) para
    construir:
      - inputs  = tokens[:-1]
      - targets = tokens[1:]

    Niveles de complejidad:
      - easy: patrones repetitivos con poco ruido.
      - medium: mezcla de motivos, dinámica tipo Markov y dependencias de copia.
      - hard: mayor mezcla de patrones, cambios de régimen y ruido.
    """

    def __init__(
        self,
        num_samples: int = 10_000,
        seq_len: int = 128,
        vocab_size: int = 1_000,
        complexity: str = "medium",
        noise_prob: float = 0.05,
        copy_prob: float = 0.35,
        regime_switch_prob: float = 0.10,
        seed: Optional[int] = None,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if complexity not in {"easy", "medium", "hard"}:
            raise ValueError("complexity must be one of: easy, medium, hard")
        if not (0.0 <= noise_prob <= 1.0):
            raise ValueError("noise_prob must be between 0.0 and 1.0")
        if not (0.0 <= copy_prob <= 1.0):
            raise ValueError("copy_prob must be between 0.0 and 1.0")
        if not (0.0 <= regime_switch_prob <= 1.0):
            raise ValueError("regime_switch_prob must be between 0.0 and 1.0")

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.complexity = complexity
        self.noise_prob = noise_prob
        self.copy_prob = copy_prob
        self.regime_switch_prob = regime_switch_prob
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

        self._tokens = self._build_tokens()

    def _rand(self, size):
        if self._generator is None:
            return torch.rand(size)
        return torch.rand(size, generator=self._generator)

    def _randint(self, low: int, high: int, size):
        if self._generator is None:
            return torch.randint(low=low, high=high, size=size, dtype=torch.long)
        return torch.randint(low=low, high=high, size=size, dtype=torch.long, generator=self._generator)

    def _randint_scalar(self, low: int, high: int) -> int:
        return int(self._randint(low, high, (1,)).item())

    def _rand_scalar(self) -> float:
        return float(self._rand((1,)).item())

    def _inject_noise(self, seq: torch.Tensor, prob: float) -> torch.Tensor:
        if prob <= 0.0:
            return seq
        noise_mask = self._rand(seq.shape) < prob
        num_noisy = int(noise_mask.sum().item())
        if num_noisy > 0:
            seq[noise_mask] = self._randint(0, self.vocab_size, (num_noisy,))
        return seq

    def _pattern_bucket(self) -> str:
        pick = self._rand_scalar()
        if self.complexity == "easy":
            if pick < 0.75:
                return "motif"
            if pick < 0.90:
                return "markov"
            return "copy"
        if self.complexity == "medium":
            if pick < 0.40:
                return "motif"
            if pick < 0.75:
                return "markov"
            return "copy"

        # hard
        if pick < 0.25:
            return "motif"
        if pick < 0.60:
            return "markov"
        return "copy"

    def _generate_motif_sequence(self, total_len: int) -> torch.Tensor:
        max_motif_len = max(4, min(total_len // 2, 24))
        motif_len = self._randint_scalar(3, max_motif_len + 1)
        motif = self._randint(0, self.vocab_size, (motif_len,))
        repeats = (total_len + motif_len - 1) // motif_len
        seq = motif.repeat(repeats)[:total_len]

        if self.complexity in {"medium", "hard"}:
            block_size = self._randint_scalar(8, min(24, total_len) + 1)
            for start in range(0, total_len, block_size):
                end = min(start + block_size, total_len)
                if self._rand_scalar() < 0.25:
                    seq[start:end] = torch.flip(seq[start:end], dims=[0])

        local_noise = self.noise_prob * (0.5 if self.complexity == "easy" else 1.0)
        return self._inject_noise(seq, local_noise)

    def _generate_markov_sequence(self, total_len: int) -> torch.Tensor:
        seq = torch.empty(total_len, dtype=torch.long)
        state = self._randint_scalar(0, self.vocab_size)
        max_stride = max(3, self.vocab_size // 100)
        stride = self._randint_scalar(1, max_stride)

        jump_prob = 0.03 if self.complexity == "easy" else (0.06 if self.complexity == "medium" else 0.10)
        jitter_high = 3 if self.complexity == "easy" else (6 if self.complexity == "medium" else 9)

        for pos in range(total_len):
            seq[pos] = state

            if self._rand_scalar() < self.regime_switch_prob:
                stride = self._randint_scalar(1, max_stride)

            if self._rand_scalar() < jump_prob:
                state = self._randint_scalar(0, self.vocab_size)
                continue

            jitter = self._randint_scalar(0, jitter_high)
            state = (state + stride + jitter) % self.vocab_size

        return self._inject_noise(seq, self.noise_prob)

    def _generate_copy_sequence(self, total_len: int) -> torch.Tensor:
        seq = self._randint(0, self.vocab_size, (total_len,))

        max_delay = max(3, min(total_len // 2, 48))
        delay = self._randint_scalar(2, max_delay)
        offset = self._randint_scalar(1, min(32, self.vocab_size))

        for pos in range(delay, total_len):
            if self._rand_scalar() < self.copy_prob:
                seq[pos] = (seq[pos - delay] + offset + (pos % 7)) % self.vocab_size

        if self.complexity == "hard":
            block_size = self._randint_scalar(10, min(40, total_len) + 1)
            topical_span = max(16, self.vocab_size // 40)
            for start in range(0, total_len, block_size):
                end = min(start + block_size, total_len)
                if self._rand_scalar() < 0.55:
                    topic_center = int(seq[start].item())
                    seq[start:end] = (topic_center + self._randint(0, topical_span, (end - start,))) % self.vocab_size

        return self._inject_noise(seq, self.noise_prob)

    def _build_tokens(self) -> torch.Tensor:
        total_len = self.seq_len + 1
        tokens = torch.empty((self.num_samples, total_len), dtype=torch.long)

        for idx in range(self.num_samples):
            pattern = self._pattern_bucket()
            if pattern == "motif":
                seq = self._generate_motif_sequence(total_len)
            elif pattern == "markov":
                seq = self._generate_markov_sequence(total_len)
            else:
                seq = self._generate_copy_sequence(total_len)
            tokens[idx] = seq

        return tokens

    def __len__(self) -> int:
        return self._tokens.size(0)

    def __getitem__(self, idx: int):
        tokens = self._tokens[idx]
        inputs = tokens[:-1]
        targets = tokens[1:]
        return inputs, targets


def create_synthetic_dataloader(
    batch_size: int = 32,
    num_samples: int = 10_000,
    seq_len: int = 128,
    vocab_size: int = 1_000,
    complexity: str = "medium",
    noise_prob: float = 0.05,
    copy_prob: float = 0.35,
    regime_switch_prob: float = 0.10,
    shuffle: bool = True,
    seed: Optional[int] = None,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    dataset = SyntheticNextTokenDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        complexity=complexity,
        noise_prob=noise_prob,
        copy_prob=copy_prob,
        regime_switch_prob=regime_switch_prob,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    loader = create_synthetic_dataloader(
        batch_size=2,
        num_samples=8,
        seq_len=16,
        vocab_size=1_000,
        complexity="hard",
        noise_prob=0.08,
        copy_prob=0.45,
        regime_switch_prob=0.20,
        seed=42,
    )

    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 16)
    assert targets.shape == (2, 16)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    print("OK - synthetic_data: datos complejos con shift correcto")
