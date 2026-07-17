"""
AI / Machine Learning — Count Trainable Parameters with Weight Tying
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""

import numpy as np


def embedding_head_parameter_count(vocab_size, hidden_size, tied=False, output_bias=True):
    """
    Count parameters for token embedding and LM output head.

    HINT:
      Untied weights: embedding V*D plus head D*V.
      Tied weights: one shared V*D matrix.
      Output bias, if present, adds V.
    """
    pass


def tied_logits(hidden, embedding_weight, output_bias=None):
    """
    Compute logits from hidden states using a tied embedding matrix.

    hidden shape: (..., hidden_size)
    embedding_weight shape: (vocab_size, hidden_size)
    """
    pass


def count_unique_trainable_parameters(parameters):
    """
    Count unique arrays by object identity.

    parameters is an iterable of (array, trainable_bool).
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_embedding_head_parameter_count():
    check(embedding_head_parameter_count(50_000, 768, tied=False, output_bias=True) == 76_850_000, "untied count wrong")
    check(embedding_head_parameter_count(50_000, 768, tied=True, output_bias=True) == 38_450_000, "tied count wrong")
    check(embedding_head_parameter_count(10, 4, tied=True, output_bias=False) == 40, "tied no-bias count wrong")
    print("PASS  embedding_head_parameter_count")


def test_tied_logits():
    hidden = np.array([[1.0, 2.0], [0.5, -1.0]])
    embedding = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    bias = np.array([0.1, 0.2, -0.5])
    logits = tied_logits(hidden, embedding, bias)
    expected = np.array([[1.1, 2.2, -0.5], [0.6, -0.8, 1.5]])
    check(np.allclose(logits, expected), f"logits wrong: {logits}")
    print("PASS  tied_logits")


def test_count_unique_trainable_parameters():
    shared = np.zeros((3, 4))
    bias = np.zeros(3)
    frozen = np.zeros((10, 10))
    params = [(shared, True), (shared, True), (bias, True), (frozen, False)]
    check(count_unique_trainable_parameters(params) == 15, "unique trainable count wrong")
    print("PASS  count_unique_trainable_parameters")


if __name__ == "__main__":
    test_embedding_head_parameter_count()
    test_tied_logits()
    test_count_unique_trainable_parameters()
    print("All tests passed.")
