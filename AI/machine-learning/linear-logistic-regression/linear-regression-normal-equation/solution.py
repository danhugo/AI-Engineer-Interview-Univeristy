"""
Reference solutions for AI / Machine Learning — Linear Regression Normal Equation.

Open this only after attempting practice.py.
"""

import numpy as np


def np_add_bias_column(X):
    """Add a leading column of 1s to X."""
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.c_[ones, X]


def np_fit_lstsq(X, y):
    """Fit linear regression with np.linalg.lstsq."""
    Xb = np_add_bias_column(X)
    y = np.asarray(y, dtype=float)
    return np.linalg.lstsq(Xb, y, rcond=None)[0]


def np_predict(X, theta):
    """Predict with theta from np_fit_lstsq."""
    Xb = np_add_bias_column(X)
    theta = np.asarray(theta, dtype=float)
    return Xb @ theta


def np_mse(y_true, y_pred):
    """Return mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def torch_add_bias_column(X):
    """Add a leading column of 1s to a PyTorch tensor."""
    import torch

    X = X.float()
    ones = torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)
    return torch.cat([ones, X], dim=1)


def torch_fit_lstsq(X, y):
    """Fit linear regression with torch.linalg.lstsq."""
    import torch

    Xb = torch_add_bias_column(X)
    y = y.float()
    return torch.linalg.lstsq(Xb, y).solution


def torch_predict(X, theta):
    """Predict with theta from torch_fit_lstsq."""
    Xb = torch_add_bias_column(X)
    return Xb @ theta.float()


# ======================================================================
# TESTS — kept identical to practice.py
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


X_REG = np.array([[1.0, 2.0],
                  [2.0, 1.0],
                  [3.0, 4.0],
                  [4.0, 3.0]])
y_REG = 2.0 * X_REG[:, 0] + 3.0 * X_REG[:, 1] + 1.0


def test_numpy():
    Xb = np_add_bias_column(X_REG)
    check(Xb.shape == (4, 3), f"bias shape wrong: {Xb.shape}")
    check(np.allclose(Xb[:, 0], 1.0), "first column should be ones")
    print("PASS  np_add_bias_column")

    theta = np_fit_lstsq(X_REG, y_REG)
    check(theta.shape == (3,), f"theta shape wrong: {theta.shape}")
    check(np.allclose(theta, [1.0, 2.0, 3.0], atol=1e-6),
          f"theta wrong: {theta}")
    print("PASS  np_fit_lstsq")

    pred = np_predict(X_REG, theta)
    check(np.allclose(pred, y_REG), f"predictions wrong: {pred}")
    print("PASS  np_predict")

    loss = np_mse(y_REG, pred)
    check(abs(float(loss)) < 1e-10, f"MSE should be near zero: {loss}")
    print("PASS  np_mse")


def _run_torch_tests():
    import torch

    X = torch.tensor(X_REG)
    y = torch.tensor(y_REG)

    Xb = torch_add_bias_column(X)
    check(tuple(Xb.shape) == (4, 3), f"bias shape wrong: {Xb.shape}")
    check(torch.allclose(Xb[:, 0], torch.ones(4)), "first column should be ones")
    print("PASS  torch_add_bias_column")

    theta = torch_fit_lstsq(X, y)
    check(tuple(theta.shape) == (3,), f"theta shape wrong: {theta.shape}")
    check(torch.allclose(theta, torch.tensor([1.0, 2.0, 3.0]), atol=1e-5),
          f"theta wrong: {theta}")
    print("PASS  torch_fit_lstsq")

    pred = torch_predict(X, theta)
    check(torch.allclose(pred, y.float(), atol=1e-5), f"predictions wrong: {pred}")
    print("PASS  torch_predict")


def run_torch_tests():
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"SKIP  PyTorch tests ({e})")
        return
    _run_torch_tests()


if __name__ == "__main__":
    test_numpy()
    run_torch_tests()
    print("\nAll tests passed.")
