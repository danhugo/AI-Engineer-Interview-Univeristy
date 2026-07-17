"""AI / Machine Learning - Speculative Acceptance vs Temperature Practice."""

import math


def softmax_with_temperature(logits, temperature):
    """Return a stable softmax over logits / temperature."""
    pass


def expected_acceptance(target_probs, draft_probs):
    """Return sum_x min(target_probs[x], draft_probs[x])."""
    pass


def accepted_prefix_length(target_tokens, draft_tokens):
    """Return how many draft tokens match the target prefix before first mismatch."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_acceptance_overlap():
    target = [0.7, 0.2, 0.1]
    draft = [0.6, 0.3, 0.1]
    check(math.isclose(expected_acceptance(target, draft), 0.9), "overlap should be 0.9")
    print("PASS  acceptance overlap")


def test_temperature_and_prefix():
    target_logits = [4.0, 1.0, 0.0]
    draft_logits = [3.0, 2.0, 0.0]

    cold = expected_acceptance(
        softmax_with_temperature(target_logits, 0.5),
        softmax_with_temperature(draft_logits, 0.5),
    )
    hot = expected_acceptance(
        softmax_with_temperature(target_logits, 2.0),
        softmax_with_temperature(draft_logits, 2.0),
    )
    check(cold > 0.0 and hot > 0.0, "acceptance should be positive")
    check(accepted_prefix_length([1, 2, 3], [1, 2, 9]) == 2, "prefix stops at mismatch")
    print("PASS  temperature and prefix")


if __name__ == "__main__":
    test_acceptance_overlap()
    test_temperature_and_prefix()
    print("All tests passed.")
