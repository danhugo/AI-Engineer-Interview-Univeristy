"""
Reference solutions for AI / Machine Learning — Ridge Regression Loss.

Open this only after attempting practice.py.
"""

import numpy as np


def np_linear_predict(X, w, b):
    """Predict y_hat = X @ w + b."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return X @ w + b


def np_mse(y_true, y_pred):
    """Return mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def np_l2_penalty(w, alpha):
    """Return alpha * sum(w ** 2)."""
    w = np.asarray(w, dtype=float)
    return alpha * np.sum(w ** 2)


def np_ridge_loss(X, y, w, b, alpha):
    """Return MSE + alpha * sum(w ** 2)."""
    pred = np_linear_predict(X, w, b)
    return np_mse(y, pred) + np_l2_penalty(w, alpha)


def np_ridge_loss_from_theta(X, y, theta, alpha):
    """Return Ridge loss when theta stores [bias, weights...]."""
    theta = np.asarray(theta, dtype=float)
    b = theta[0]
    w = theta[1:]
    return np_ridge_loss(X, y, w, b, alpha)


def torch_ridge_loss(pred, y, weight, alpha):
    """Return MSE + alpha * sum(weight ** 2)."""
    import torch

    mse = torch.nn.MSELoss()(pred, y.float())
    penalty = alpha * torch.sum(weight ** 2)
    return mse + penalty


def torch_ridge_loss_from_model(model, X, y, alpha):
    """Return Ridge loss for a torch.nn.Linear model."""
    pred = model(X.float())
    return torch_ridge_loss(pred, y, model.weight, alpha)


# ======================================================================
# TESTS — kept identical to practice.py
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
