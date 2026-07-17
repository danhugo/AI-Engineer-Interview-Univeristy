"""
AI / Machine Learning - KV Cache Tiered Offloading
PRACTICE FILE

HOW TO USE
1. Replace each `pass` with your implementation.
2. Run: python practice.py
3. Open solution.py only after you try.
"""


DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int8": 1,
}


def kv_cache_bytes(num_layers, batch_size, sequence_length, hidden_size, dtype="float16"):
    """Estimate decoder-only KV cache bytes."""
    pass


def split_gpu_cpu(total_bytes, gpu_budget_bytes):
    """Return (gpu_bytes, cpu_bytes) after keeping as much as possible on GPU."""
    pass


def transfer_time_ms(num_bytes, bandwidth_gb_per_s):
    """Return ideal transfer time in milliseconds using decimal GB/s."""
    pass


def can_hide_prefetch(num_bytes, bandwidth_gb_per_s, available_compute_ms):
    """Return True if ideal transfer time fits before the data is needed."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_kv_cache_bytes():
    total = kv_cache_bytes(32, 2, 1024, 4096, "float16")
    check(total == 1_073_741_824, f"KV bytes wrong: {total}")
    print("PASS  kv_cache_bytes")


def test_split_gpu_cpu():
    check(split_gpu_cpu(1000, 600) == (600, 400), "split with offload wrong")
    check(split_gpu_cpu(1000, 2000) == (1000, 0), "split without offload wrong")
    print("PASS  split_gpu_cpu")


def test_transfer_time_ms():
    check(round(transfer_time_ms(1_000_000_000, 25), 1) == 40.0, "transfer time wrong")
    print("PASS  transfer_time_ms")


def test_can_hide_prefetch():
    check(can_hide_prefetch(250_000_000, 25, 10.0), "prefetch should fit")
    check(not can_hide_prefetch(500_000_000, 25, 10.0), "prefetch should not fit")
    print("PASS  can_hide_prefetch")


if __name__ == "__main__":
    test_kv_cache_bytes()
    test_split_gpu_cpu()
    test_transfer_time_ms()
    test_can_hide_prefetch()
    print("All tests passed.")
