"""Reference solutions for Adam / AdamW Optimizer Step."""

import numpy as np


def np_adam_step(theta, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Return (theta_new, m_new, v_new) after one Adam step."""
    theta = np.asarray(theta, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    theta_new = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta_new, m_new, v_new


def np_adamw_step(theta, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999,
                  eps=1e-8, weight_decay=0.01):
    """Return (theta_new, m_new, v_new) after one AdamW step."""
    theta = np.asarray(theta, dtype=float)
    decayed_theta = theta - lr * weight_decay * theta
    adam_theta, m_new, v_new = np_adam_step(
        decayed_theta, grad, m, v, t, lr=lr, beta1=beta1, beta2=beta2, eps=eps
    )
    return adam_theta, m_new, v_new


def np_fit_quadratic_with_adam(theta0, lr=0.1, steps=80):
    """Minimize f(theta) = sum((theta - target)^2) with Adam."""
    theta = np.asarray(theta0, dtype=float)
    target = np.array([2.0, -1.0])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        grad = 2 * (theta - target)
        theta, m, v = np_adam_step(theta, grad, m, v, t, lr=lr)
    return theta


def torch_one_adamw_step(param, grad, lr=0.1, weight_decay=0.01):
    """Apply one PyTorch AdamW step to a tensor parameter and return it."""
    import torch

    optimizer = torch.optim.AdamW([param], lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    param.grad = grad.clone()
    optimizer.step()
    return param


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
