"""AI / Machine Learning — Linear Warmup Schedule Practice."""


def linear_warmup_lr(step, base_lr, warmup_steps):
    """
    Return learning rate for linear warmup then constant base_lr.

    HINT:
      During warmup use base_lr * step / warmup_steps.
      After warmup return base_lr.
    """
    pass


def linear_warmup_multipliers(total_steps, warmup_steps):
    """
    Return LR multipliers for steps 0..total_steps-1.
    """
    pass


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
