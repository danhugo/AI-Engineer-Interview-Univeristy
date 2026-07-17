"""
AI / Machine Learning - LoRA Adapter Hot-Swap Overhead
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


def lora_layer_params(in_features, out_features, rank):
    """Return LoRA parameters for one adapted linear layer."""
    pass


def total_lora_params(layers):
    """
    Return total LoRA parameters.

    layers contains tuples: (in_features, out_features, rank).
    """
    pass


def adapter_bytes(layers, dtype="float16"):
    """Return adapter parameter bytes for the provided adapted layers."""
    pass


def cold_swap_time_ms(adapter_num_bytes, bandwidth_gb_per_s, fixed_overhead_ms=0.0):
    """Return ideal transfer time plus fixed overhead in milliseconds."""
    pass


def cache_hit_rate(total_requests, cold_swaps):
    """Return the fraction of requests whose adapter was already resident."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_lora_layer_params():
    check(lora_layer_params(4096, 4096, 8) == 65_536, "single layer params wrong")
    print("PASS  lora_layer_params")


def test_total_lora_params_and_bytes():
    layers = [(4096, 4096, 8), (4096, 11008, 8)]
    check(total_lora_params(layers) == 186_368, "total params wrong")
    check(adapter_bytes(layers, "float16") == 372_736, "adapter bytes wrong")
    print("PASS  total_lora_params_and_bytes")


def test_cold_swap_time_ms():
    ms = cold_swap_time_ms(1_000_000_000, bandwidth_gb_per_s=25, fixed_overhead_ms=3)
    check(round(ms, 1) == 43.0, f"cold swap time wrong: {ms}")
    print("PASS  cold_swap_time_ms")


def test_cache_hit_rate():
    check(cache_hit_rate(100, 7) == 0.93, "hit rate wrong")
    check(cache_hit_rate(0, 0) == 0.0, "empty hit rate wrong")
    print("PASS  cache_hit_rate")


if __name__ == "__main__":
    test_lora_layer_params()
    test_total_lora_params_and_bytes()
    test_cold_swap_time_ms()
    test_cache_hit_rate()
    print("All tests passed.")
