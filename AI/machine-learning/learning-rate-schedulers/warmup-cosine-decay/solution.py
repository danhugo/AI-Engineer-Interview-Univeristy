"""Reference solutions for Warmup + Cosine Decay."""

import math


def warmup_cosine_lr(step, base_lr, total_steps, warmup_steps, min_lr=0.0):
    """Return LR for linear warmup followed by cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps

    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def warmup_cosine_schedule(total_steps, base_lr, warmup_steps, min_lr=0.0):
    """Return list of learning rates for all steps."""
    return [warmup_cosine_lr(step, base_lr, total_steps, warmup_steps, min_lr) for step in range(total_steps)]


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
