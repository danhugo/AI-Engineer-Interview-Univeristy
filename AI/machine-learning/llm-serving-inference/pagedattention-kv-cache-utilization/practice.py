"""
AI / Machine Learning - PagedAttention KV Cache Utilization
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""


def blocks_needed(tokens, block_size):
    """Return the number of physical KV blocks needed for one sequence."""
    pass


def allocated_token_slots(tokens, block_size):
    """Return allocated token slots for one sequence."""
    pass


def wasted_token_slots(tokens, block_size):
    """Return unused token slots in the final block."""
    pass


def sequence_utilization(tokens, block_size):
    """Return used token slots divided by allocated token slots."""
    pass


def batch_utilization(sequence_lengths, block_size):
    """Return aggregate KV slot utilization for a batch."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_blocks_needed():
    check(blocks_needed(0, 16) == 0, "zero tokens wrong")
    check(blocks_needed(1, 16) == 1, "one token wrong")
    check(blocks_needed(32, 16) == 2, "exact blocks wrong")
    check(blocks_needed(33, 16) == 3, "partial block wrong")
    print("PASS  blocks_needed")


def test_slots_and_waste():
    check(allocated_token_slots(33, 16) == 48, "allocated slots wrong")
    check(wasted_token_slots(33, 16) == 15, "wasted slots wrong")
    check(wasted_token_slots(32, 16) == 0, "exact waste wrong")
    print("PASS  slots_and_waste")


def test_sequence_utilization():
    check(sequence_utilization(0, 16) == 1.0, "empty utilization wrong")
    check(round(sequence_utilization(33, 16), 4) == 0.6875, "sequence utilization wrong")
    print("PASS  sequence_utilization")


def test_batch_utilization():
    util = batch_utilization([16, 17, 31], 16)
    check(round(util, 4) == round(64 / 80, 4), f"batch utilization wrong: {util}")
    print("PASS  batch_utilization")


if __name__ == "__main__":
    test_blocks_needed()
    test_slots_and_waste()
    test_sequence_utilization()
    test_batch_utilization()
    print("All tests passed.")
