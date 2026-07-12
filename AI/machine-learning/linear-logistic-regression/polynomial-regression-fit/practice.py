"""
============================================================
  AI / Machine Learning — Polynomial Regression Fit
  PRACTICE FILE
  Write every function yourself. Tests tell you if you got
  it right. Do NOT open solution.py first.
============================================================

HOW TO USE
----------
1. Read the hint for each function.
2. Delete the `pass` and write your code.
3. Run:  python practice.py
4. A test PASS means your logic is correct. Fix until all pass.
5. Only open solution.py after you finish or are truly stuck.

Requirements:  pip install numpy torch
   (If torch is missing, the PyTorch tests print SKIP instead
    of failing — but install it to really practice PyTorch.)
"""

import numpy as np


# ======================================================================
# LEVEL 1 — NumPy
# ======================================================================

def np_polynomial_features(x, degree):
    """
    Build a polynomial feature matrix [1, x, x^2, ..., x^degree].

    HINT:
      1. Convert x to a 1-D NumPy array.
      2. Build one column for each power from 0 through degree.
      3. Use np.column_stack.
    """
    # TODO
    pass


def np_fit_polynomial_regression(x, y, degree):
    """
    Fit polynomial regression with np.linalg.lstsq.
    Return theta where theta[0] is the bias term.

    HINT:
      1. Phi = np_polynomial_features(x, degree)
      2. theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
    """
    # TODO
    pass


def np_predict_polynomial(x, theta):
    """
    Predict using fitted polynomial coefficients.

    HINT:
      1. degree = len(theta) - 1
      2. Phi = np_polynomial_features(x, degree)
      3. return Phi @ theta
    """
    # TODO
    pass


def np_mse(y_true, y_pred):
    """
    Return mean squared error.
    """
    # TODO
    pass


# ======================================================================
# LEVEL 2 — PyTorch
# ======================================================================

def torch_polynomial_features(x, degree):
    """
    Build a polynomial feature matrix with PyTorch.

    HINT:
      Use torch.stack([x ** power for power in range(degree + 1)], dim=1).
    """
    # TODO
    pass


def torch_fit_polynomial_regression(x, y, degree):
    """
    Fit polynomial regression with torch.linalg.lstsq.
    Return theta.
    """
    # TODO
    pass


def torch_predict_polynomial(x, theta):
    """
    Predict using fitted polynomial coefficients in PyTorch.
    """
    # TODO
    pass


# ======================================================================
# TESTS — do not edit
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

