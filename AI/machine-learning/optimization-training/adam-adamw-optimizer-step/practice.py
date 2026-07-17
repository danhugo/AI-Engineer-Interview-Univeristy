"""
AI / Machine Learning — Adam / AdamW Optimizer Step
PRACTICE FILE
"""

import numpy as np


def np_adam_step(theta, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Return (theta_new, m_new, v_new) after one Adam step.

    HINT:
      1. Update m and v.
      2. Bias-correct them with t.
      3. Subtract lr * m_hat / (sqrt(v_hat) + eps).
    """
    pass


def np_adamw_step(theta, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999,
                  eps=1e-8, weight_decay=0.01):
    """
    Return (theta_new, m_new, v_new) after one AdamW step.

    HINT:
      Apply decoupled weight decay directly to theta, then the Adam update.
    """
    pass


def np_fit_quadratic_with_adam(theta0, lr=0.1, steps=80):
    """
    Minimize f(theta) = sum((theta - target)^2) with Adam.

    HINT:
      gradient = 2 * (theta - target)
    """
    pass


def torch_one_adamw_step(param, grad, lr=0.1, weight_decay=0.01):
    """
    Apply one PyTorch AdamW step to a tensor parameter and return it.

    HINT:
      Set param.grad, create torch.optim.AdamW([param]), then step.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_numpy_adam():
    theta = np.array([1.0, -2.0])
    grad = np.array([0.1, -0.2])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    theta1, m1, v1 = np_adam_step(theta, grad, m, v, t=1, lr=0.1)
    check(np.allclose(m1, [0.01, -0.02]), f"m wrong: {m1}")
    check(np.allclose(v1, [0.00001, 0.00004]), f"v wrong: {v1}")
    check(np.allclose(theta1, [0.90000001, -1.900000005]), f"theta wrong: {theta1}")
    print("PASS  np_adam_step")


def test_numpy_adamw():
    theta = np.array([1.0, -2.0])
    grad = np.array([0.1, -0.2])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    theta1, _, _ = np_adamw_step(theta, grad, m, v, t=1, lr=0.1, weight_decay=0.1)
    check(np.allclose(theta1, [0.89000001, -1.880000005]), f"adamw theta wrong: {theta1}")
    print("PASS  np_adamw_step")

    trained = np_fit_quadratic_with_adam(np.array([5.0, -5.0]), lr=0.1, steps=120)
    check(np.allclose(trained, [2.0, -1.0], atol=0.2), f"Adam should approach target: {trained}")
    print("PASS  np_fit_quadratic_with_adam")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return

    param = torch.tensor([1.0, -2.0], requires_grad=True)
    grad = torch.tensor([0.1, -0.2])
    result = torch_one_adamw_step(param, grad, lr=0.1, weight_decay=0.1)
    expected = torch.tensor([0.89, -1.88])
    check(torch.allclose(result.detach(), expected, atol=1e-6), f"torch AdamW wrong: {result}")
    print("PASS  torch_one_adamw_step")


if __name__ == "__main__":
    test_numpy_adam()
    test_numpy_adamw()
    test_torch()
    print("All tests passed.")
