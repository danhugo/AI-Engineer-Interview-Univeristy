"""Turning logits into a token.

Everything until now was greedy argmax, which is enough to verify correctness
but not enough for an API — `temperature` and `top_p` are part of the OpenAI
request shape.

Sampling params are per-sequence, and a batched step mixes sequences with
different settings, so they arrive as tensors with one row each.

Tensor-parallel note: our ranks stay in lockstep only because they all derive
the same token. Random sampling would break that, so the generator is seeded
from a step counter shared by every rank — same seed, same logits, same draw.
"""

import torch


class Sampler:
    def __init__(self, seed: int = 0, device: str = "cuda"):
        self.generator = torch.Generator(device=device)
        self.base_seed = seed
        self.step = 0

    def __call__(self, logits: torch.Tensor, temperature: torch.Tensor,
                 top_p: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        """logits (B, V), params (B,). Returns (B,) token ids."""
        greedy = logits.argmax(dim=-1)
        if bool((temperature <= 0).all()):
            return greedy

        # Reseed per step so every TP rank draws identically.
        self.generator.manual_seed(self.base_seed + self.step)
        self.step += 1

        scaled = logits.float() / temperature.clamp(min=1e-5).unsqueeze(1)

        if bool((top_k > 0).any()):
            scaled = self._mask_top_k(scaled, top_k)
        if bool((top_p < 1).any()):
            scaled = self._mask_top_p(scaled, top_p)

        probs = scaled.softmax(dim=-1)
        drawn = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
        # temperature == 0 means "be deterministic", so keep argmax for those.
        return torch.where(temperature > 0, drawn, greedy)

    @staticmethod
    def _mask_top_k(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        """Keep each row's k best logits, drop the rest."""
        vocab = logits.shape[-1]
        k = top_k.clamp(min=1, max=vocab)
        # A row with top_k <= 0 means "no limit", so let it keep everything.
        k = torch.where(top_k > 0, k, torch.full_like(k, vocab))
        sorted_logits, _ = logits.sort(dim=-1, descending=True)
        # The k-th best value per row becomes the cutoff.
        kth = sorted_logits.gather(1, (k - 1).unsqueeze(1).long())
        return logits.masked_fill(logits < kth, float("-inf"))

    @staticmethod
    def _mask_top_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """Nucleus sampling: keep the smallest set of tokens reaching mass p."""
        sorted_logits, idx = logits.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Drop tokens once the mass BEFORE them already reached p, so the token
        # that crosses the threshold is kept and the top-1 always survives.
        drop = (cumulative - sorted_logits.softmax(dim=-1)) >= top_p.unsqueeze(1)
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        return sorted_logits.scatter(1, idx, sorted_logits)
