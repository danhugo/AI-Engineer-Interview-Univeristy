"""Reference solutions for CosineAnnealingLR."""

import math


def cosine_annealing_lr(step, base_lr, t_max, eta_min=0.0):
    """Return cosine annealing learning rate."""
    step = min(step, t_max)
    return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * step / t_max))


def cosine_annealing_schedule(total_steps, base_lr, t_max, eta_min=0.0):
    """Return list of cosine annealing learning rates."""
    return [cosine_annealing_lr(step, base_lr, t_max, eta_min) for step in range(total_steps)]


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def test_schedule():
    vals = cosine_annealing_schedule(5, base_lr=0.1, t_max=4, eta_min=0.0)
    expected = [0.1, 0.08535533905932738, 0.05, 0.014644660940672627, 0.0]
    check(all(close(a, b) for a, b in zip(vals, expected)), f"cosine schedule wrong: {vals}")
    print("PASS  CosineAnnealingLR")


if __name__ == "__main__":
    test_schedule()
    print("All tests passed.")
