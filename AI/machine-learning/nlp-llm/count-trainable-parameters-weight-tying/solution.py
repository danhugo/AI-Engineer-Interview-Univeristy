"""Reference solutions for Count Trainable Parameters with Weight Tying."""

import numpy as np


def embedding_head_parameter_count(vocab_size, hidden_size, tied=False, output_bias=True):
    """Count parameters for token embedding and LM output head."""
    if tied:
        count = vocab_size * hidden_size
    else:
        count = vocab_size * hidden_size + hidden_size * vocab_size
    if output_bias:
        count += vocab_size
    return count


def tied_logits(hidden, embedding_weight, output_bias=None):
    """Compute logits from hidden states using a tied embedding matrix."""
    hidden = np.asarray(hidden, dtype=float)
    embedding_weight = np.asarray(embedding_weight, dtype=float)
    logits = hidden @ embedding_weight.T
    if output_bias is not None:
        logits = logits + np.asarray(output_bias, dtype=float)
    return logits


def count_unique_trainable_parameters(parameters):
    """Count unique arrays by object identity."""
    seen = set()
    total = 0
    for array, trainable in parameters:
        if not trainable or id(array) in seen:
            continue
        seen.add(id(array))
        total += np.asarray(array).size
    return total


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
