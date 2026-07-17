"""AI / Machine Learning - Disaggregated Prefill-Decode Serving Practice."""


def kv_cache_bytes(num_layers, prompt_tokens, hidden_size, bytes_per_element):
    """Return simplified KV cache bytes for one request."""
    pass


def ttft(prefill_queue_ms, prefill_ms, kv_transfer_ms, first_decode_ms):
    """Return time to first token in milliseconds."""
    pass


def worker_utilization(arrival_rate_per_s, service_ms, num_workers):
    """Return offered utilization for a worker pool."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_latency_decomposition():
    check(ttft(5, 40, 8, 12) == 65, "TTFT should sum prefill path components")
    check(kv_cache_bytes(2, 3, 4, 2) == 96, "KV cache formula should include K and V")
    print("PASS  latency decomposition")


def test_utilization():
    prefill_util = worker_utilization(arrival_rate_per_s=10, service_ms=50, num_workers=2)
    decode_util = worker_utilization(arrival_rate_per_s=10, service_ms=200, num_workers=4)

    check(prefill_util == 0.25, "prefill utilization should be 0.25")
    check(decode_util == 0.5, "decode utilization should be 0.5")
    print("PASS  utilization")


if __name__ == "__main__":
    test_latency_decomposition()
    test_utilization()
    print("All tests passed.")
