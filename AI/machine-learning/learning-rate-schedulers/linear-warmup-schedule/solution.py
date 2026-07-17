"""Reference solutions for Linear Warmup Schedule."""


def linear_warmup_lr(step, base_lr, warmup_steps):
    """Return learning rate for linear warmup then constant base_lr."""
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


def linear_warmup_multipliers(total_steps, warmup_steps):
    """Return LR multipliers for steps 0..total_steps-1."""
    return [linear_warmup_lr(step, 1.0, warmup_steps) for step in range(total_steps)]


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def test_schedule():
    vals = [linear_warmup_lr(s, base_lr=0.1, warmup_steps=4) for s in range(7)]
    expected = [0.0, 0.025, 0.05, 0.075, 0.1, 0.1, 0.1]
    check(all(close(a, b) for a, b in zip(vals, expected)), f"warmup lr wrong: {vals}")
    mult = linear_warmup_multipliers(5, warmup_steps=4)
    check(all(close(a, b) for a, b in zip(mult, [0.0, 0.25, 0.5, 0.75, 1.0])), f"multipliers wrong: {mult}")
    print("PASS  schedule")


if __name__ == "__main__":
    test_schedule()
    print("All tests passed.")
