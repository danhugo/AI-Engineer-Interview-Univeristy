"""
Tests for the from-scratch Transformer in trasnformer.py.

Run:
    pytest test_trasnformer.py
    # or, from this folder:
    pytest
"""

import torch
import pytest

from trasnformer import Embedding


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _make(vocab=10, dim=4, padding_idx=0, seed=0):
    """Build a deterministic Embedding so tests are reproducible."""
    torch.manual_seed(seed)
    return Embedding(vocab, dim, padding_idx=padding_idx)


class TestEmbeddingForward:
    """Shape and lookup behavior of Embedding.forward."""

    def test_output_shape(self):
        emb = _make()
        x = torch.tensor([[0, 1, 2], [3, 4, 5]])  # (batch=2, seq_len=3)
        out = emb(x)
        assert out.shape == (2, 3, 4)

    def test_lookup_matches_weight_rows(self):
        # The embedding of token id i must equal weight[i].
        emb = _make()
        x = torch.tensor([[0, 1, 2]])
        out = emb(x)
        for col, token_id in enumerate([0, 1, 2]):
            assert torch.equal(out[0, col], emb.weight[token_id])

    def test_supports_1d_input(self):
        # A single un-batched sequence still works.
        emb = _make()
        x = torch.tensor([0, 1, 2])
        assert emb(x).shape == (3, 4)

    def test_out_of_range_token_raises(self):
        # Token id >= vocab_size is an invalid index.
        emb = _make(vocab=5)
        with pytest.raises(IndexError):
            emb(torch.tensor([5]))


class TestEmbeddingPadding:
    """padding_idx must keep the padding row at zero and frozen."""

    def test_padding_row_starts_zero(self):
        emb = _make(padding_idx=0)
        assert torch.all(emb.weight[0] == 0)

    def test_padding_lookup_is_zero_vector(self):
        emb = _make(padding_idx=0)
        out = emb(torch.tensor([[0, 1]]))
        assert torch.all(out[0, 0] == 0)   # padding position
        assert not torch.all(out[0, 1] == 0)  # real token is nonzero

    def test_padding_row_grad_is_zero_after_backward(self):
        # The whole point of the gradient hook.
        emb = _make(padding_idx=0)
        out = emb(torch.tensor([[0, 1, 2, 0, 3]]))  # includes padding
        out.sum().backward()
        assert emb.weight.grad is not None
        assert torch.all(emb.weight.grad[0] == 0)  # padding row untouched

    def test_non_padding_rows_get_grad(self):
        emb = _make(padding_idx=0)
        out = emb(torch.tensor([[1, 2, 3]]))
        out.sum().backward()
        assert torch.any(emb.weight.grad[1:] != 0)

    def test_padding_row_stays_zero_after_optimizer_step(self):
        emb = _make(padding_idx=0)
        out = emb(torch.tensor([[0, 1, 2]]))
        out.sum().backward()
        torch.optim.SGD(emb.parameters(), lr=1.0).step()
        assert torch.all(emb.weight[0] == 0)

    def test_no_padding_idx_no_freezing(self):
        # Without padding_idx, every row is free to learn.
        emb = _make(padding_idx=None)
        out = emb(torch.tensor([[0, 1, 2]]))
        out.sum().backward()
        assert torch.any(emb.weight.grad[0] != 0)


class TestEmbeddingInit:
    """Weight shape and initialization."""

    def test_weight_shape(self):
        emb = _make(vocab=10, dim=4)
        assert emb.weight.shape == (10, 4)

    def test_weights_are_learnable(self):
        emb = _make()
        assert emb.weight.requires_grad
