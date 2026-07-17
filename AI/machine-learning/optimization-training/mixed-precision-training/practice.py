"""
AI / Machine Learning - Mixed Precision Training
PRACTICE FILE
"""

import numpy as np


def quantize_activation(x, dtype=np.float16):
    """
    Simulate storing an activation in lower precision, then returning it as float32.

    HINT:
      Convert to dtype, then convert back to np.float32.
    """
    pass


def mse_loss_and_grad(w, X, y, loss_scale=1.0):
    """
    Return (scaled_loss, scaled_gradient) for linear regression MSE.

    HINT:
      pred = X @ w
      loss = mean((pred - y) ** 2)
      grad = (2 / n) * X.T @ (pred - y)
      Return both multiplied by loss_scale.
    """
    pass


def unscale_gradient(scaled_grad, loss_scale):
    """
    Return the real gradient after loss scaling.

    HINT:
      Divide by loss_scale.
    """
    pass


def sgd_master_weight_step(w_master, grad, lr):
    """
    Return a float32 master-weight SGD update.

    HINT:
      Convert w_master and grad to float32 before subtracting lr * grad.
    """
    pass


def detect_overflow_or_nan(arrays):
    """
    Return True if any array contains NaN or infinity.

    HINT:
      np.isfinite(array).all() is useful.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_quantize_activation():
    x = np.array([1.25, 2.5, 3.75], dtype=np.float32)
    q = quantize_activation(x)
    check(q.dtype == np.float32, f"returned dtype should be float32, got {q.dtype}")
    check(np.allclose(q, x), f"simple values should survive fp16: {q}")
    print("PASS  quantize_activation")


def test_loss_scaling_math():
    X = np.array([[1.0, 0.0],
                  [1.0, 1.0],
                  [1.0, 2.0]], dtype=np.float32)
    y = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    w = np.array([0.0, 0.0], dtype=np.float32)

    scaled_loss, scaled_grad = mse_loss_and_grad(w, X, y, loss_scale=128.0)
    real_grad = unscale_gradient(scaled_grad, 128.0)

    check(abs(float(scaled_loss) - 128.0 * (35.0 / 3.0)) < 1e-4,
          f"scaled loss wrong: {scaled_loss}")
    check(np.allclose(real_grad, [-6.0, -26.0 / 3.0]), f"unscaled grad wrong: {real_grad}")
    print("PASS  loss scaling math")


def test_master_weight_update_and_overflow():
    w = np.array([1.0, -1.0], dtype=np.float16)
    grad = np.array([0.1, -0.2], dtype=np.float16)
    updated = sgd_master_weight_step(w, grad, lr=0.5)
    check(updated.dtype == np.float32, f"master weights should be float32: {updated.dtype}")
    check(np.allclose(updated, [0.9500122, -0.9000244]), f"update wrong: {updated}")

    check(detect_overflow_or_nan([np.array([1.0, np.inf])]), "should detect infinity")
    check(not detect_overflow_or_nan([np.array([1.0, 2.0])]), "finite values should pass")
    print("PASS  master update / overflow detection")


if __name__ == "__main__":
    test_quantize_activation()
    test_loss_scaling_math()
    test_master_weight_update_and_overflow()
    print("All tests passed.")
