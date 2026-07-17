"""Reference solutions for Dropout."""

import numpy as np


def inverted_dropout(x, p, rng, training=True):
    """Return inverted dropout output."""
    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1")

    x = np.asarray(x, dtype=float)
    if not training or p == 0:
        return x.copy()

    keep_prob = 1 - p
    mask = rng.random(x.shape) >= p
    return (x * mask) / keep_prob


def dropout_expected_value(x, p):
    """Return the theoretical expected output of inverted dropout."""
    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1")
    return np.asarray(x, dtype=float)


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_eval_is_identity():
    x = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    out = inverted_dropout(x, p=0.5, rng=rng, training=False)
    check(np.allclose(out, x), "eval dropout should be identity")
    print("PASS  eval_is_identity")


def test_train_mask_and_scale():
    x = np.ones(6)
    rng = np.random.default_rng(7)
    out = inverted_dropout(x, p=0.5, rng=rng, training=True)
    expected = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 2.0])
    check(np.allclose(out, expected), f"dropout mask or scaling wrong: {out}")
    print("PASS  train_mask_and_scale")


def test_expected_value():
    x = np.array([2.0, -4.0, 0.5])
    check(np.allclose(dropout_expected_value(x, p=0.25), x), "expected value should equal x")
    print("PASS  expected_value")


def test_invalid_p():
    rng = np.random.default_rng(0)
    for p in [-0.1, 1.0, 1.2]:
        try:
            inverted_dropout(np.ones(2), p=p, rng=rng, training=True)
        except ValueError:
            continue
        raise AssertionError(f"FAIL  p={p} should raise ValueError")
    print("PASS  invalid_p")


if __name__ == "__main__":
    test_eval_is_identity()
    test_train_mask_and_scale()
    test_expected_value()
    test_invalid_p()
    print("All tests passed.")
