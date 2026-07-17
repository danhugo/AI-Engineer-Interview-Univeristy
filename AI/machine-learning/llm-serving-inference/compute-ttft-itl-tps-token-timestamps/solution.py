"""Reference solutions for Compute TTFT, ITL, TPS from Token Timestamps."""


def time_to_first_token(request_start_s, token_times_s):
    """Return TTFT in seconds."""
    if not token_times_s:
        raise ValueError("token_times_s must contain at least one token timestamp")
    return token_times_s[0] - request_start_s


def inter_token_latencies(token_times_s):
    """Return gaps between consecutive token timestamps."""
    return [curr - prev for prev, curr in zip(token_times_s, token_times_s[1:])]


def mean_itl(token_times_s):
    """Return average ITL, or None if fewer than 2 tokens exist."""
    gaps = inter_token_latencies(token_times_s)
    if not gaps:
        return None
    return sum(gaps) / len(gaps)


def system_tps(request_starts_s, response_token_times_s):
    """
    Return aggregate output TPS.

    response_token_times_s is a list of lists, one timestamp list per request.
    """
    if not request_starts_s:
        raise ValueError("request_starts_s must not be empty")
    all_token_times = [t for times in response_token_times_s for t in times]
    if not all_token_times:
        raise ValueError("at least one output token timestamp is required")
    elapsed = max(all_token_times) - min(request_starts_s)
    if elapsed <= 0:
        raise ValueError("measurement window must be positive")
    return len(all_token_times) / elapsed


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
