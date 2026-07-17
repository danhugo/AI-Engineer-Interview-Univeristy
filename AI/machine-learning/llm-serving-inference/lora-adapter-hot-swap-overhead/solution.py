"""Reference solutions for LoRA Adapter Hot-Swap Overhead."""


DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int8": 1,
}


def lora_layer_params(in_features, out_features, rank):
    """Return LoRA parameters for one adapted linear layer."""
    return rank * (in_features + out_features)


def total_lora_params(layers):
    """Return total LoRA parameters for (in_features, out_features, rank) tuples."""
    return sum(lora_layer_params(in_features, out_features, rank) for in_features, out_features, rank in layers)


def adapter_bytes(layers, dtype="float16"):
    """Return adapter parameter bytes for the provided adapted layers."""
    return total_lora_params(layers) * DTYPE_BYTES[dtype]


def cold_swap_time_ms(adapter_num_bytes, bandwidth_gb_per_s, fixed_overhead_ms=0.0):
    """Return ideal transfer time plus fixed overhead in milliseconds."""
    bytes_per_second = bandwidth_gb_per_s * 1_000_000_000
    transfer_ms = (adapter_num_bytes / bytes_per_second) * 1000
    return transfer_ms + fixed_overhead_ms


def cache_hit_rate(total_requests, cold_swaps):
    """Return the fraction of requests whose adapter was already resident."""
    if total_requests == 0:
        return 0.0
    return (total_requests - cold_swaps) / total_requests


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
