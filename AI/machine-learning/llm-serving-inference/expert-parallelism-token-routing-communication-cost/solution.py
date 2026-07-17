"""Reference solutions for Expert Parallel Communication Cost."""


def count_assignments(routes, num_experts):
    """Return a list with how many token assignments each expert receives."""
    counts = [0] * num_experts
    for token_routes in routes:
        for expert in token_routes:
            counts[expert] += 1
    return counts


def remote_assignment_fraction(routes, expert_to_gpu, source_gpu):
    """Return the fraction of assignments whose expert is not on source_gpu."""
    total = 0
    remote = 0
    for token_routes in routes:
        for expert in token_routes:
            total += 1
            if expert_to_gpu[expert] != source_gpu:
                remote += 1
    return 0.0 if total == 0 else remote / total


def estimate_all_to_all_bytes(num_tokens, top_k, hidden_size, bytes_per_element, remote_fraction):
    """Estimate dispatch plus combine bytes for routed expert activations."""
    dispatch = num_tokens * top_k * hidden_size * bytes_per_element * remote_fraction
    return int(2 * dispatch)


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_routing_counts():
    routes = [[0, 2], [2, 3], [0, 3]]
    check(count_assignments(routes, 4) == [2, 0, 2, 2], "expert counts should match routes")
    print("PASS  routing counts")


def test_communication_estimate():
    routes = [[0, 2], [2, 3]]
    expert_to_gpu = {0: 0, 1: 0, 2: 1, 3: 1}
    remote = remote_assignment_fraction(routes, expert_to_gpu, source_gpu=0)
    check(remote == 0.75, "three of four assignments are remote")

    bytes_total = estimate_all_to_all_bytes(2, 2, 4, 2, remote)
    check(bytes_total == 48, "dispatch plus combine bytes should be 48")
    print("PASS  communication estimate")


if __name__ == "__main__":
    test_routing_counts()
    test_communication_estimate()
    print("All tests passed.")
