"""AI / Machine Learning - Temperature Sampling Practice."""

import numpy as np


def softmax_with_temperature(logits, temperature):
    """Return softmax(logits / temperature) using a stable implementation."""
    pass


def entropy(probs):
    """Return Shannon entropy of a probability vector."""
    pass


def sample_token(probs, rng):
    """Return one sampled index from a categorical distribution."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_temperature_shapes_distribution():
    logits = np.array([4.0, 2.0, 0.0])
    cold = softmax_with_temperature(logits, 0.5)
    base = softmax_with_temperature(logits, 1.0)
    hot = softmax_with_temperature(logits, 2.0)

    check(np.allclose(base.sum(), 1.0), "probabilities must sum to 1")
    check(cold[0] > base[0] > hot[0], "top token probability should fall as temperature rises")
    check(entropy(cold) < entropy(base) < entropy(hot), "entropy should rise with temperature")
    print("PASS  temperature distribution")


def test_sampling_and_validation():
    rng = np.random.default_rng(7)
    probs = np.array([0.0, 0.0, 1.0])
    check(sample_token(probs, rng) == 2, "only nonzero-probability token should be sampled")

    try:
        softmax_with_temperature([1.0, 2.0], 0.0)
    except ValueError:
        print("PASS  validation")
        return
    raise AssertionError("FAIL  non-positive temperature should raise ValueError")


if __name__ == "__main__":
    test_temperature_shapes_distribution()
    test_sampling_and_validation()
    print("All tests passed.")
