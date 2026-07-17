"""Reference solutions for Cosine Annealing with Warm Restarts."""

import math


def warm_restart_cycle_position(step, t_0, t_mult=1):
    """Return (t_cur, t_i) for the current cycle."""
    t_i = t_0
    remaining = step
    while remaining >= t_i:
        remaining -= t_i
        t_i *= t_mult
    return remaining, t_i


def cosine_warm_restart_lr(step, base_lr, t_0, t_mult=1, eta_min=0.0):
    """Return cosine annealing warm restart learning rate."""
    t_cur, t_i = warm_restart_cycle_position(step, t_0, t_mult)
    return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t_cur / t_i))


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def test_schedule():
    vals = [cosine_warm_restart_lr(s, 0.1, t_0=2, t_mult=2) for s in range(7)]
    expected = [0.1, 0.05, 0.1, 0.08535533905932738, 0.05, 0.014644660940672627, 0.1]
    check(all(close(a, b) for a, b in zip(vals, expected)), f"warm restart wrong: {vals}")
    print("PASS  CosineAnnealingWarmRestarts")


if __name__ == "__main__":
    test_schedule()
    print("All tests passed.")
