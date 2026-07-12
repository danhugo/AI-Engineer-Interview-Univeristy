"""
============================================================
  AI / Machine Learning — Ridge Regression Loss
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

Ridge regularizes weights, not the bias.
"""

import numpy as np


def np_linear_predict(X, w, b):
    """
    Predict y_hat = X @ w + b.

    HINT:
      Convert X and w to NumPy arrays with dtype=float.
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


def np_l2_penalty(w, alpha):
    """
    Return alpha * sum(w ** 2).

    HINT:
      Only pass feature weights into this function, not the bias.
    """
    # TODO
    pass


def np_ridge_loss(X, y, w, b, alpha):
    """
    Return MSE + alpha * sum(w ** 2).

    HINT:
      1. pred = np_linear_predict(X, w, b)
      2. return np_mse(y, pred) + np_l2_penalty(w, alpha)
    """
    # TODO
    pass


def np_ridge_loss_from_theta(X, y, theta, alpha):
    """
    Return Ridge loss when theta stores [bias, weights...].

    HINT:
      1. b = theta[0]
      2. w = theta[1:]
      3. Do not regularize theta[0].
    """
    # TODO
    pass


def torch_ridge_loss(pred, y, weight, alpha):
    """
    Return MSE + alpha * sum(weight ** 2).

    HINT:
      1. import torch
      2. mse = torch.nn.MSELoss()(pred, y.float())
      3. penalty = alpha * torch.sum(weight ** 2)
    """
    # TODO
    pass


def torch_ridge_loss_from_model(model, X, y, alpha):
    """
    Return Ridge loss for a torch.nn.Linear model.

    HINT:
      1. pred = model(X.float())
      2. Regularize model.weight only.
      3. Do not regularize model.bias.
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
    if hasattr(a, "detach"):
        a = a.detach()
    return abs(float(a) - float(b)) < tol


X_REG = np.array([[1.0, 2.0],
                  [2.0, 1.0],
                  [3.0, 0.0]])
w_REG = np.array([2.0, -1.0])
b_REG = 0.5
y_REG = np_linear_predict(X_REG, w_REG, b_REG)


def test_numpy():
    pred = np_linear_predict(X_REG, w_REG, b_REG)
    check(np.allclose(pred, [0.5, 3.5, 6.5]), f"predictions wrong: {pred}")
    print("PASS  np_linear_predict")

    check(close(np_mse(y_REG, pred), 0.0), "MSE should be zero for exact labels")
    print("PASS  np_mse")

    penalty = np_l2_penalty(w_REG, alpha=0.1)
    check(close(penalty, 0.5), f"penalty wrong: {penalty}")
    print("PASS  np_l2_penalty")

    loss = np_ridge_loss(X_REG, y_REG, w_REG, b_REG, alpha=0.1)
    check(close(loss, 0.5), f"ridge loss wrong: {loss}")
    print("PASS  np_ridge_loss")

    theta = np.array([100.0, 2.0, -1.0])
    noisy_y = np_linear_predict(X_REG, w_REG, b=100.0)
    theta_loss = np_ridge_loss_from_theta(X_REG, noisy_y, theta, alpha=0.1)
    check(close(theta_loss, 0.5),
          f"bias should not be regularized, got loss: {theta_loss}")
    print("PASS  np_ridge_loss_from_theta")


def _run_torch_tests():
    import torch

    pred = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[1.0], [2.0], [4.0]])
    weight = torch.tensor([[2.0, -1.0]])

    loss = torch_ridge_loss(pred, y, weight, alpha=0.1)
    expected = (1.0 / 3.0) + 0.5
    check(close(loss, expected), f"torch ridge loss wrong: {loss}")
    print("PASS  torch_ridge_loss")

    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.bias.copy_(torch.tensor([100.0]))

    X = torch.tensor(X_REG)
    y_model = model(X.float()).detach()
    model_loss = torch_ridge_loss_from_model(model, X, y_model, alpha=0.1)
    check(close(model_loss, 0.5),
          f"model bias should not be regularized, got loss: {model_loss}")
    print("PASS  torch_ridge_loss_from_model")


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
