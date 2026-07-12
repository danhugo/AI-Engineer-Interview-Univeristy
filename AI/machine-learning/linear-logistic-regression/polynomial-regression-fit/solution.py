"""
Reference solutions for AI / Machine Learning — Polynomial Regression Fit.

Open this only after attempting practice.py.
"""

import numpy as np


def np_polynomial_features(x, degree):
    """Build a polynomial feature matrix [1, x, x^2, ..., x^degree]."""
    x = np.asarray(x, dtype=float).reshape(-1)
    columns = [x ** power for power in range(degree + 1)]
    return np.column_stack(columns)


def np_fit_polynomial_regression(x, y, degree):
    """Fit polynomial regression with np.linalg.lstsq."""
    Phi = np_polynomial_features(x, degree)
    y = np.asarray(y, dtype=float)
    return np.linalg.lstsq(Phi, y, rcond=None)[0]


def np_predict_polynomial(x, theta):
    """Predict using fitted polynomial coefficients."""
    theta = np.asarray(theta, dtype=float)
    degree = len(theta) - 1
    Phi = np_polynomial_features(x, degree)
    return Phi @ theta


def np_mse(y_true, y_pred):
    """Return mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def torch_polynomial_features(x, degree):
    """Build a polynomial feature matrix with PyTorch."""
    import torch

    x = x.float().reshape(-1)
    columns = [x ** power for power in range(degree + 1)]
    return torch.stack(columns, dim=1)


def torch_fit_polynomial_regression(x, y, degree):
    """Fit polynomial regression with torch.linalg.lstsq."""
    import torch

    Phi = torch_polynomial_features(x, degree)
    y = y.float().reshape(-1)
    return torch.linalg.lstsq(Phi, y).solution


def torch_predict_polynomial(x, theta):
    """Predict using fitted polynomial coefficients in PyTorch."""
    degree = len(theta) - 1
    Phi = torch_polynomial_features(x, degree)
    return Phi @ theta.float()


# ======================================================================
# TESTS — kept identical to practice.py
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


x_DATA = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
y_DATA = 1.0 + 2.0 * x_DATA + 3.0 * x_DATA ** 2


def test_numpy():
    Phi = np_polynomial_features(np.array([0.0, 1.0, 2.0]), degree=3)
    expected = np.array([[1.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 2.0, 4.0, 8.0]])
    check(Phi.shape == (3, 4), f"feature shape wrong: {Phi.shape}")
    check(np.allclose(Phi, expected), f"features wrong: {Phi}")
    print("PASS  np_polynomial_features")

    theta = np_fit_polynomial_regression(x_DATA, y_DATA, degree=2)
    check(theta.shape == (3,), f"theta shape wrong: {theta.shape}")
    check(np.allclose(theta, [1.0, 2.0, 3.0], atol=1e-6), f"theta wrong: {theta}")
    print("PASS  np_fit_polynomial_regression")

    pred = np_predict_polynomial(x_DATA, theta)
    check(np.allclose(pred, y_DATA, atol=1e-6), f"predictions wrong: {pred}")
    print("PASS  np_predict_polynomial")

    check(close(np_mse(y_DATA, pred), 0.0), "MSE should be zero for exact quadratic")
    print("PASS  np_mse")


def _run_torch_tests():
    import torch

    x = torch.tensor([0.0, 1.0, 2.0])
    Phi = torch_polynomial_features(x, degree=3)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0, 1.0],
                             [1.0, 2.0, 4.0, 8.0]])
    check(torch.allclose(Phi, expected), f"torch features wrong: {Phi}")
    print("PASS  torch_polynomial_features")

    x_train = torch.tensor(x_DATA, dtype=torch.float32)
    y_train = torch.tensor(y_DATA, dtype=torch.float32)
    theta = torch_fit_polynomial_regression(x_train, y_train, degree=2)
    check(torch.allclose(theta, torch.tensor([1.0, 2.0, 3.0]), atol=1e-5), f"torch theta wrong: {theta}")
    print("PASS  torch_fit_polynomial_regression")

    pred = torch_predict_polynomial(x_train, theta)
    check(torch.allclose(pred, y_train, atol=1e-5), f"torch predictions wrong: {pred}")
    print("PASS  torch_predict_polynomial")


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

