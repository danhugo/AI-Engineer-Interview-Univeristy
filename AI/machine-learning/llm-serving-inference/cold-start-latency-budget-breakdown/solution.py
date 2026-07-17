"""Reference solutions for Cold Start Latency Budget Breakdown."""


def segment_durations(timestamps_s):
    """
    Return adjacent segment durations from ordered (name, timestamp) pairs.

    Example output key: "start->ready".
    """
    durations = {}
    for (left_name, left_t), (right_name, right_t) in zip(timestamps_s, timestamps_s[1:]):
        durations[f"{left_name}->{right_name}"] = right_t - left_t
    return durations


def total_cold_start(timestamps_s):
    """Return time from first timestamp to last timestamp."""
    if len(timestamps_s) < 2:
        raise ValueError("at least two timestamps are required")
    return timestamps_s[-1][1] - timestamps_s[0][1]


def warmup_savings(cold_first_request_s, warm_first_request_s):
    """Return absolute latency saved by warming."""
    return cold_first_request_s - warm_first_request_s


def within_budget(total_s, budget_s):
    """Return True if total_s is less than or equal to budget_s."""
    return total_s <= budget_s


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
