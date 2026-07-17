"""
AI / Machine Learning - Continuous Batching Throughput
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""


def request_level_slots_used(output_lengths):
    """
    Return total occupied slot-iterations under fixed request-level batching.

    All requests occupy a slot until the longest request finishes.
    """
    pass


def continuous_slots_used(output_lengths):
    """Return useful slot-iterations when slots free as each request finishes."""
    pass


def wasted_slot_iterations(output_lengths):
    """Return request-level occupied slots minus useful generated-token slots."""
    pass


def decode_tps(active_sequences, step_time_s):
    """Approximate generated tokens per second for one decode iteration shape."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_request_level_slots_used():
    check(request_level_slots_used([2, 5, 3]) == 15, "request-level slots wrong")
    print("PASS  request_level_slots_used")


def test_continuous_slots_used():
    check(continuous_slots_used([2, 5, 3]) == 10, "continuous useful slots wrong")
    print("PASS  continuous_slots_used")


def test_wasted_slot_iterations():
    check(wasted_slot_iterations([2, 5, 3]) == 5, "wasted slots wrong")
    print("PASS  wasted_slot_iterations")


def test_decode_tps():
    check(decode_tps(32, 0.08) == 400.0, "decode TPS wrong")
    print("PASS  decode_tps")


if __name__ == "__main__":
    test_request_level_slots_used()
    test_continuous_slots_used()
    test_wasted_slot_iterations()
    test_decode_tps()
    print("All tests passed.")
