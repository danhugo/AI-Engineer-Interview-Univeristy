"""Reference solutions for Tokens Per Second Throughput."""


def aggregate_tps(total_tokens, start_time_s, end_time_s):
    """Return aggregate tokens per second for a measurement window."""
    elapsed = end_time_s - start_time_s
    if elapsed <= 0:
        raise ValueError("end_time_s must be greater than start_time_s")
    return total_tokens / elapsed


def per_user_tps(output_tokens, e2e_latency_s):
    """Return one request's output-token rate."""
    if e2e_latency_s <= 0:
        raise ValueError("e2e_latency_s must be positive")
    return output_tokens / e2e_latency_s


def split_token_throughput(prompt_tokens, generation_tokens, elapsed_s):
    """Return (prompt_tps, generation_tps)."""
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive")
    return prompt_tokens / elapsed_s, generation_tokens / elapsed_s


def capacity_utilization(measured_tps, theoretical_peak_tps):
    """Return utilization as a fraction from 0 to 1."""
    if theoretical_peak_tps <= 0:
        raise ValueError("theoretical_peak_tps must be positive")
    return measured_tps / theoretical_peak_tps


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
