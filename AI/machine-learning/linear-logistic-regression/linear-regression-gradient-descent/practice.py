"""
============================================================
  AI / Machine Learning — Linear Regression Gradient Descent
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


def np_add_bias_column(X):
    """
    Add a leading column of 1s to X.

    HINT:
      Convert X to float, create a column of ones, then concatenate.
    """
    # TODO
    pass


def np_predict(X, theta):
    """
    Predict y_hat using a bias-column parameter vector.

    HINT:
      Add the bias column, then return Xb @ theta.
    """
    # TODO
    pass


def np_mse(y_true, y_pred):
    """
    Return mean squared error.

    HINT:
      Convert both inputs to NumPy arrays and average squared errors.
    """
    # TODO
    pass


def np_mse_gradient(X, y, theta):
    """
    Return the vectorized gradient of MSE for theta.

    HINT:
      1. Xb = np_add_bias_column(X)
      2. pred = Xb @ theta
      3. grad = (2 / n) * Xb.T @ (pred - y)
    """
    # TODO
    pass


def np_gradient_descent_step(X, y, theta, lr):
    """
    Return theta after one gradient-descent update.

    HINT:
      theta_new = theta - lr * np_mse_gradient(X, y, theta)
    """
    # TODO
    pass


def np_fit_linear_gradient_descent(X, y, lr=0.05, steps=2000):
    """
    Fit linear regression with vectorized gradient descent.
    Return theta where theta[0] is bias and theta[1:] are weights.

    HINT:
      Start theta as zeros and repeatedly call np_gradient_descent_step.
    """
    # TODO
    pass


def torch_train_linear_model(X, y, lr=0.05, steps=500):
    """
    Train torch.nn.Linear with MSELoss and an optimizer.
    Return the trained model.

    HINT:
      1. import torch
      2. model = torch.nn.Linear(X.shape[1], 1)
      3. optimizer = torch.optim.SGD(model.parameters(), lr=lr)
      4. Repeat: zero_grad, predict, loss, backward, step.
    """
    # TODO
    pass


def torch_predict(model, X):
    """
    Return model predictions without tracking gradients.

    HINT:
      Use `with torch.no_grad(): return model(X.float())`.
    """
    # TODO
    pass


# ======================================================================
# TESTS — do not edit
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
