"""AI / Machine Learning — Warmup + Cosine Decay Practice."""

import math


def warmup_cosine_lr(step, base_lr, total_steps, warmup_steps, min_lr=0.0):
    """
    Return LR for linear warmup followed by cosine decay.

    HINT:
      Warmup: base_lr * step / warmup_steps.
      Decay: cosine from base_lr to min_lr.
    """
    pass


def warmup_cosine_schedule(total_steps, base_lr, warmup_steps, min_lr=0.0):
    """Return list of learning rates for all steps."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def test_schedule():
    vals = warmup_cosine_schedule(total_steps=6, base_lr=0.1, warmup_steps=2, min_lr=0.0)
    expected = [0.0, 0.05, 0.1, 0.08535533905932738, 0.05, 0.014644660940672627]
    check(all(close(a, b) for a, b in zip(vals, expected)), f"schedule wrong: {vals}")
    print("PASS  schedule")


if __name__ == "__main__":
    test_schedule()
    print("All tests passed.")
