"""AI / Machine Learning - API vs Dedicated GPU Break-Even Practice."""


def api_cost(input_tokens, output_tokens, input_price_per_million, output_price_per_million):
    """Return API cost in dollars."""
    pass


def gpu_cost_per_million_tokens(hourly_cost, tokens_per_second, utilization):
    """Return dedicated GPU cost per million generated billable tokens."""
    pass


def break_even_tokens_per_second(hourly_cost, api_price_per_million):
    """Return sustained tokens/sec needed for GPU hourly cost to match API price."""
    pass


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_api_cost():
    cost = api_cost(2_000_000, 500_000, 1.0, 6.0)
    check(cost == 5.0, "API cost should separate input and output prices")
    print("PASS  API cost")


def test_gpu_break_even():
    cost_per_m = gpu_cost_per_million_tokens(hourly_cost=3.6, tokens_per_second=1000, utilization=0.5)
    check(cost_per_m == 2.0, "GPU cost per million should include utilization")

    rate = break_even_tokens_per_second(hourly_cost=3.6, api_price_per_million=2.0)
    check(rate == 500.0, "break-even rate should be 500 tokens/sec")
    print("PASS  GPU break-even")


if __name__ == "__main__":
    test_api_cost()
    test_gpu_break_even()
    print("All tests passed.")
