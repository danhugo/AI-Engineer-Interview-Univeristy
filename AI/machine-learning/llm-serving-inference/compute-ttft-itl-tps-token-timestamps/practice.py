"""
AI / Machine Learning - Compute TTFT, ITL, TPS from Token Timestamps
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""


def time_to_first_token(request_start_s, token_times_s):
    """Return TTFT in seconds."""
    pass


def inter_token_latencies(token_times_s):
    """Return gaps between consecutive token timestamps."""
    pass


def mean_itl(token_times_s):
    """Return average ITL, or None if fewer than 2 tokens exist."""
    pass


def system_tps(request_starts_s, response_token_times_s):
    """
    Return aggregate output TPS.

    response_token_times_s is a list of lists, one timestamp list per request.
    """
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_time_to_first_token():
    check(round(time_to_first_token(10.0, [10.35, 10.55]), 3) == 0.35, "TTFT wrong")
    print("PASS  time_to_first_token")


def test_inter_token_latencies():
    gaps = [round(gap, 3) for gap in inter_token_latencies([1.0, 1.2, 1.7])]
    check(gaps == [0.2, 0.5], "ITL gaps wrong")
    print("PASS  inter_token_latencies")


def test_mean_itl():
    check(round(mean_itl([1.0, 1.2, 1.7]), 3) == 0.35, "mean ITL wrong")
    check(mean_itl([1.0]) is None, "single-token ITL should be None")
    print("PASS  mean_itl")


def test_system_tps():
    starts = [0.0, 0.5]
    token_times = [[1.0, 1.5, 2.0], [1.25, 1.75]]
    check(round(system_tps(starts, token_times), 3) == 2.5, "system TPS wrong")
    print("PASS  system_tps")


if __name__ == "__main__":
    test_time_to_first_token()
    test_inter_token_latencies()
    test_mean_itl()
    test_system_tps()
    print("All tests passed.")
