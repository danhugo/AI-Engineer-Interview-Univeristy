"""AI / Machine Learning — StepLR Practice."""


def step_lr(step, base_lr, step_size, gamma):
    """Return StepLR learning rate at step."""
    pass


def step_lr_schedule(total_steps, base_lr, step_size, gamma):
    """Return list of StepLR learning rates."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_schedule():
    vals = step_lr_schedule(7, base_lr=0.1, step_size=3, gamma=0.1)
    expected = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001]
    check(all(abs(a - b) < 1e-12 for a, b in zip(vals, expected)), f"StepLR wrong: {vals}")
    print("PASS  StepLR")


if __name__ == "__main__":
    test_schedule()
    print("All tests passed.")
