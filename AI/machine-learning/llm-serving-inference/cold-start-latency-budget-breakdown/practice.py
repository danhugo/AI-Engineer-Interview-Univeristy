"""
AI / Machine Learning - Cold Start Latency Budget Breakdown
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""


def segment_durations(timestamps_s):
    """
    Return adjacent segment durations from ordered (name, timestamp) pairs.

    Example output key: "start->ready".
    """
    pass


def total_cold_start(timestamps_s):
    """Return time from first timestamp to last timestamp."""
    pass


def warmup_savings(cold_first_request_s, warm_first_request_s):
    """Return absolute latency saved by warming."""
    pass


def within_budget(total_s, budget_s):
    """Return True if total_s is less than or equal to budget_s."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_segment_durations():
    ts = [("start", 0.0), ("loaded", 12.0), ("ready", 20.0)]
    check(segment_durations(ts) == {"start->loaded": 12.0, "loaded->ready": 8.0}, "segments wrong")
    print("PASS  segment_durations")


def test_total_cold_start():
    ts = [("start", 5.0), ("ready", 42.5)]
    check(total_cold_start(ts) == 37.5, "total cold start wrong")
    print("PASS  total_cold_start")


def test_warmup_savings():
    check(warmup_savings(7.5, 1.2) == 6.3, "warmup savings wrong")
    print("PASS  warmup_savings")


def test_within_budget():
    check(within_budget(29.9, 30.0), "should be within budget")
    check(not within_budget(31.0, 30.0), "should exceed budget")
    print("PASS  within_budget")


if __name__ == "__main__":
    test_segment_durations()
    test_total_cold_start()
    test_warmup_savings()
    test_within_budget()
    print("All tests passed.")
