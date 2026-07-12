"""
============================================================
  AI / Machine Learning — Linear Regression Normal Equation
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

Use NumPy and PyTorch solvers. Do not compute a matrix inverse.
"""

import numpy as np


def np_add_bias_column(X):
    """
    Add a leading column of 1s to X.

    HINT:
      1. Convert X to a NumPy array with dtype=float.
      2. Create ones with shape (rows, 1).
      3. Use np.c_[ones, X] or np.concatenate.
    """
    # TODO
    pass


def np_fit_lstsq(X, y):
    """
    Fit linear regression with np.linalg.lstsq.
    Return theta where theta[0] is bias and theta[1:] are weights.

    HINT:
      1. Xb = np_add_bias_column(X)
      2. Convert y to a float NumPy array.
      3. Return np.linalg.lstsq(Xb, y, rcond=None)[0].
    """
    # TODO
    pass


def np_predict(X, theta):
    """
    Predict with theta from np_fit_lstsq.

    HINT:
      Add the bias column, then return Xb @ theta.
    """
    # TODO
    pass


def np_mse(y_true, y_pred):
    """
    Return mean squared error.

    HINT:
      Use np.mean((y_pred - y_true) ** 2).
    """
    # TODO
    pass


def torch_add_bias_column(X):
    """
    Add a leading column of 1s to a PyTorch tensor.

    HINT:
      1. import torch
      2. Convert X to float.
      3. Create ones on the same device as X.
      4. Use torch.cat([ones, X], dim=1).
    """
    # TODO
    pass


def torch_fit_lstsq(X, y):
    """
    Fit linear regression with torch.linalg.lstsq.
    Return theta where theta[0] is bias and theta[1:] are weights.

    HINT:
      1. Xb = torch_add_bias_column(X)
      2. Convert y to float.
      3. Return torch.linalg.lstsq(Xb, y).solution.
    """
    # TODO
    pass


def torch_predict(X, theta):
    """
    Predict with theta from torch_fit_lstsq.

    HINT:
      Add the bias column, then return Xb @ theta.
    """
    # TODO
    pass


# ======================================================================
# TESTS — do not edit
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
