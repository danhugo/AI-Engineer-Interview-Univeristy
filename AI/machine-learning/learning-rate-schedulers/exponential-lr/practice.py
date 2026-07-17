"""AI / Machine Learning — ExponentialLR Practice."""


def exponential_lr(step, base_lr, gamma):
    """Return ExponentialLR learning rate at step."""
    pass


def exponential_lr_schedule(total_steps, base_lr, gamma):
    """Return list of ExponentialLR learning rates."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_schedule():
    vals = exponential_lr_schedule(5, base_lr=0.1, gamma=0.9)
    expected = [0.1, 0.09, 0.081, 0.0729, 0.06561]
    check(all(abs(a - b) < 1e-12 for a, b in zip(vals, expected)), f"ExponentialLR wrong: {vals}")
    print("PASS  ExponentialLR")


if __name__ == "__main__":
    test_schedule()
    print("All tests passed.")
