"""Reference solutions for End-to-End Latency Decomposition."""


def sum_components(components_s):
    """Return total latency from a dict of component seconds."""
    return sum(components_s.values())


def streaming_e2e(ttft_s, output_tokens, mean_itl_s, finish_overhead_s=0.0):
    """Estimate streaming end-to-end latency."""
    if output_tokens <= 0:
        return finish_overhead_s
    return ttft_s + max(0, output_tokens - 1) * mean_itl_s + finish_overhead_s


def component_percentages(components_s):
    """Return each component's fraction of total latency."""
    total = sum_components(components_s)
    if total <= 0:
        raise ValueError("total latency must be positive")
    return {name: value / total for name, value in components_s.items()}


def dominant_component(components_s):
    """Return the component name with the largest latency."""
    if not components_s:
        raise ValueError("components_s must not be empty")
    return max(components_s, key=components_s.get)


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_sum_components():
    parts = {"queue": 0.1, "prefill": 0.4, "decode": 0.5}
    check(round(sum_components(parts), 3) == 1.0, "sum wrong")
    print("PASS  sum_components")


def test_streaming_e2e():
    check(round(streaming_e2e(0.8, 6, 0.1, 0.05), 3) == 1.35, "streaming E2E wrong")
    print("PASS  streaming_e2e")


def test_component_percentages():
    pct = component_percentages({"queue": 1.0, "prefill": 3.0})
    check(round(pct["queue"], 2) == 0.25, "queue percentage wrong")
    check(round(pct["prefill"], 2) == 0.75, "prefill percentage wrong")
    print("PASS  component_percentages")


def test_dominant_component():
    check(dominant_component({"queue": 0.2, "decode": 1.3}) == "decode", "dominant wrong")
    print("PASS  dominant_component")


if __name__ == "__main__":
    test_sum_components()
    test_streaming_e2e()
    test_component_percentages()
    test_dominant_component()
    print("All tests passed.")
