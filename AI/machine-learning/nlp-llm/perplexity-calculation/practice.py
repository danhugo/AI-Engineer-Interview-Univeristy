"""AI / Machine Learning - Perplexity Calculation Practice."""

import math
import numpy as np


def average_negative_log_likelihood(token_probs, eps=1e-12):
    """Return mean -log(p) for true-token probabilities."""
    pass


def perplexity_from_probs(token_probs):
    """Return perplexity from true-token probabilities."""
    pass


def log_softmax(logits, axis=-1):
    """Return numerically stable log-softmax."""
    pass


def sequence_cross_entropy_from_logits(logits, targets, ignore_index=-100):
    """Return mean NLL over non-ignored targets."""
    pass


def perplexity_from_logits(logits, targets, ignore_index=-100):
    """Return perplexity from logits and target token ids."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


def test_probs():
    probs = np.array([0.5, 0.25, 0.125])
    avg_nll = average_negative_log_likelihood(probs)
    ppl = perplexity_from_probs(probs)
    check(close(avg_nll, -np.mean(np.log(probs))), f"avg NLL wrong: {avg_nll}")
    check(close(ppl, math.exp(avg_nll)), f"PPL wrong: {ppl}")
    print("PASS  probabilities")


def test_logits():
    logits = np.array([
        [[2.0, 0.0, -1.0],
         [0.0, 1.0, 2.0]],
        [[1.0, 3.0, 0.0],
         [5.0, 0.0, 0.0]],
    ])
    targets = np.array([
        [0, 2],
        [1, -100],
    ])
    ce = sequence_cross_entropy_from_logits(logits, targets)
    expected = 0.2490993345
    check(close(ce, expected, tol=1e-6), f"cross entropy wrong: {ce}")
    check(close(perplexity_from_logits(logits, targets), math.exp(expected)), "PPL from logits wrong")
    print("PASS  logits and mask")


def test_stability():
    logits = np.array([[1000.0, 999.0]])
    logs = log_softmax(logits)
    check(np.isfinite(logs).all(), f"log_softmax should be finite: {logs}")
    check(close(float(np.exp(logs).sum()), 1.0), "softmax probabilities should sum to 1")
    print("PASS  stability")


if __name__ == "__main__":
    test_probs()
    test_logits()
    test_stability()
    print("All tests passed.")
