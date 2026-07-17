"""Reference solutions for Mixed Precision Training."""

import numpy as np


def quantize_activation(x, dtype=np.float16):
    """Simulate storing an activation in lower precision, then returning it as float32."""
    return np.asarray(x).astype(dtype).astype(np.float32)


def mse_loss_and_grad(w, X, y, loss_scale=1.0):
    """Return (scaled_loss, scaled_gradient) for linear regression MSE."""
    w = np.asarray(w, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    pred = X @ w
    error = pred - y
    loss = np.mean(error ** 2)
    grad = (2.0 / X.shape[0]) * (X.T @ error)
    return loss * loss_scale, grad * loss_scale


def unscale_gradient(scaled_grad, loss_scale):
    """Return the real gradient after loss scaling."""
    return np.asarray(scaled_grad, dtype=np.float32) / float(loss_scale)


def sgd_master_weight_step(w_master, grad, lr):
    """Return a float32 master-weight SGD update."""
    w_master = np.asarray(w_master, dtype=np.float32)
    grad = np.asarray(grad, dtype=np.float32)
    return w_master - float(lr) * grad


def detect_overflow_or_nan(arrays):
    """Return True if any array contains NaN or infinity."""
    return any(not np.isfinite(np.asarray(array)).all() for array in arrays)


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
