"""
Reference solutions for AI / Machine Learning — Linear Regression Gradient Descent.

Open this only after attempting practice.py.
"""

import numpy as np


def np_add_bias_column(X):
    """Add a leading column of 1s to X."""
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.c_[ones, X]


def np_predict(X, theta):
    """Predict y_hat using a bias-column parameter vector."""
    Xb = np_add_bias_column(X)
    theta = np.asarray(theta, dtype=float)
    return Xb @ theta


def np_mse(y_true, y_pred):
    """Return mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def np_mse_gradient(X, y, theta):
    """Return the vectorized gradient of MSE for theta."""
    Xb = np_add_bias_column(X)
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    pred = Xb @ theta
    n = Xb.shape[0]
    return (2 / n) * Xb.T @ (pred - y)


def np_gradient_descent_step(X, y, theta, lr):
    """Return theta after one gradient-descent update."""
    theta = np.asarray(theta, dtype=float)
    return theta - lr * np_mse_gradient(X, y, theta)


def np_fit_linear_gradient_descent(X, y, lr=0.05, steps=2000):
    """Fit linear regression with vectorized gradient descent."""
    X = np.asarray(X, dtype=float)
    theta = np.zeros(X.shape[1] + 1, dtype=float)
    for _ in range(steps):
        theta = np_gradient_descent_step(X, y, theta, lr)
    return theta


def torch_train_linear_model(X, y, lr=0.05, steps=500):
    """Train torch.nn.Linear with MSELoss and an optimizer."""
    import torch

    X = X.float()
    y = y.float()
    model = torch.nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    return model


def torch_predict(model, X):
    """Return model predictions without tracking gradients."""
    import torch

    with torch.no_grad():
        return model(X.float())


# ======================================================================
# TESTS — kept identical to practice.py
# ======================================================================

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


X_REG = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0],
                  [2.0, 1.0],
                  [1.0, 2.0]])
y_REG = 1.0 + 2.0 * X_REG[:, 0] - 3.0 * X_REG[:, 1]


def test_numpy():
    theta0 = np.zeros(3)
    grad = np_mse_gradient(X_REG, y_REG, theta0)
    check(grad.shape == (3,), f"gradient shape wrong: {grad.shape}")
    expected_grad = [-1.0 / 3.0, -4.0 / 3.0, 2.0]
    check(np.allclose(grad, expected_grad), f"gradient wrong: {grad}")
    print("PASS  np_mse_gradient")

    theta1 = np_gradient_descent_step(X_REG, y_REG, theta0, lr=0.1)
    expected_theta1 = [1.0 / 30.0, 2.0 / 15.0, -0.2]
    check(np.allclose(theta1, expected_theta1), f"one step wrong: {theta1}")
    print("PASS  np_gradient_descent_step")

    theta = np_fit_linear_gradient_descent(X_REG, y_REG, lr=0.05, steps=3000)
    pred = np_predict(X_REG, theta)
    check(np.allclose(pred, y_REG, atol=1e-2),
          f"trained predictions wrong: {pred}")
    check(float(np_mse(y_REG, pred)) < 1e-4, "MSE should be very small")
    print("PASS  np_fit_linear_gradient_descent")


def _run_torch_tests():
    import torch

    torch.manual_seed(0)
    X = torch.tensor(X_REG)
    y = torch.tensor(y_REG).view(-1, 1)

    model = torch_train_linear_model(X, y, lr=0.05, steps=1000)
    pred = torch_predict(model, X)
    check(tuple(pred.shape) == (6, 1), f"prediction shape wrong: {pred.shape}")
    check(torch.allclose(pred, y.float(), atol=2e-2),
          f"trained predictions wrong: {pred.view(-1)}")
    print("PASS  torch_train_linear_model / torch_predict")


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
