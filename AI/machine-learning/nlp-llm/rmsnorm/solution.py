"""Reference solutions for RMSNorm."""

import numpy as np


def rms(x, eps=1e-8):
    """Return RMS over the last axis, keeping dimensions for broadcasting."""
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)


def rmsnorm_forward(X, gamma, eps=1e-8):
    """Apply RMSNorm over the last axis."""
    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    return gamma * X / rms(X, eps=eps)


def layer_norm_without_affine(X, eps=1e-8):
    """Apply plain LayerNorm without gamma/beta for comparison."""
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_rms_values():
    X = np.array([[[3.0, 4.0], [0.0, 0.0]]])
    out = rms(X, eps=0.0)
    expected = np.array([[[3.53553391], [0.0]]])
    check(np.allclose(out, expected), f"rms wrong: {out}")
    print("PASS  rms_values")


def test_rmsnorm_forward():
    X = np.array([[[3.0, 4.0]]])
    gamma = np.array([1.0, 2.0])
    out = rmsnorm_forward(X, gamma, eps=0.0)
    expected = np.array([[[0.84852814, 2.2627417]]])
    check(np.allclose(out, expected), f"rmsnorm wrong: {out}")
    print("PASS  rmsnorm_forward")


def test_rmsnorm_does_not_center():
    X = np.array([[[1.0, 2.0, 3.0]]])
    rn = rmsnorm_forward(X, np.ones(3), eps=0.0)
    ln = layer_norm_without_affine(X, eps=0.0)
    check(not np.allclose(np.mean(rn, axis=-1), 0.0), "RMSNorm should not force zero mean")
    check(np.allclose(np.mean(ln, axis=-1), 0.0), "LayerNorm comparison should center")
    print("PASS  rmsnorm_does_not_center")


if __name__ == "__main__":
    test_rms_values()
    test_rmsnorm_forward()
    test_rmsnorm_does_not_center()
    print("All tests passed.")
