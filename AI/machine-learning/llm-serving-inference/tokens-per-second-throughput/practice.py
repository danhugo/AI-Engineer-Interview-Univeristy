"""
AI / Machine Learning - Tokens Per Second Throughput
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""


def aggregate_tps(total_tokens, start_time_s, end_time_s):
    """Return aggregate tokens per second for a measurement window."""
    pass


def per_user_tps(output_tokens, e2e_latency_s):
    """Return one request's output-token rate."""
    pass


def split_token_throughput(prompt_tokens, generation_tokens, elapsed_s):
    """Return (prompt_tps, generation_tps)."""
    pass


def capacity_utilization(measured_tps, theoretical_peak_tps):
    """Return utilization as a fraction from 0 to 1."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_aggregate_tps():
    check(aggregate_tps(1_000, 10.0, 20.0) == 100.0, "aggregate TPS wrong")
    print("PASS  aggregate_tps")


def test_per_user_tps():
    check(per_user_tps(120, 6.0) == 20.0, "per-user TPS wrong")
    print("PASS  per_user_tps")


def test_split_token_throughput():
    prompt_tps, generation_tps = split_token_throughput(800, 200, 4.0)
    check(prompt_tps == 200.0, "prompt TPS wrong")
    check(generation_tps == 50.0, "generation TPS wrong")
    print("PASS  split_token_throughput")


def test_capacity_utilization():
    check(round(capacity_utilization(750, 1000), 3) == 0.75, "utilization wrong")
    print("PASS  capacity_utilization")


if __name__ == "__main__":
    test_aggregate_tps()
    test_per_user_tps()
    test_split_token_throughput()
    test_capacity_utilization()
    print("All tests passed.")
