"""Reference solutions for Beam Search Decoding."""


def normalized_score(log_score, length, alpha):
    """Return length-normalized score."""
    if alpha == 0.0:
        return log_score
    return log_score / (max(length, 1) ** alpha)


def beam_search(next_log_probs_fn, start_token, end_token, beam_width, max_steps, alpha=0.0):
    """
    Return the best token sequence found by beam search.

    next_log_probs_fn(prefix) returns a dict: token -> log_probability.
    The returned sequence should exclude start_token and end_token.
    """
    beams = [([start_token], 0.0, False)]

    for _ in range(max_steps):
        candidates = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue

            for token, log_prob in next_log_probs_fn(tokens).items():
                next_tokens = tokens + [token]
                candidates.append((next_tokens, score + log_prob, token == end_token))

        candidates.sort(
            key=lambda item: normalized_score(item[1], len(item[0]) - 1, alpha),
            reverse=True,
        )
        beams = candidates[:beam_width]
        if all(finished for _, _, finished in beams):
            break

    best_tokens, _, _ = max(
        beams,
        key=lambda item: normalized_score(item[1], len(item[0]) - 1, alpha),
    )
    return [token for token in best_tokens[1:] if token != end_token]


def greedy_decode(next_log_probs_fn, start_token, end_token, max_steps):
    """Return greedy decoding result, excluding start_token and end_token."""
    tokens = [start_token]
    for _ in range(max_steps):
        next_probs = next_log_probs_fn(tokens)
        next_token = max(next_probs, key=next_probs.get)
        if next_token == end_token:
            break
        tokens.append(next_token)
    return tokens[1:]


def toy_next_log_probs(prefix):
    """Toy model where beam search finds a better sequence than greedy."""
    last = prefix[-1]
    table = {
        "<s>": {"A": -0.10, "B": -0.30},
        "A": {"</s>": -2.00, "x": -2.20},
        "B": {"C": -0.10, "</s>": -2.00},
        "C": {"</s>": -0.10},
        "x": {"</s>": -0.10},
    }
    return table.get(last, {"</s>": 0.0})


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_beam_search():
    greedy = greedy_decode(toy_next_log_probs, "<s>", "</s>", max_steps=3)
    beam = beam_search(toy_next_log_probs, "<s>", "</s>", beam_width=2, max_steps=3)
    check(greedy == ["A"], f"greedy should pick local best A: {greedy}")
    check(beam == ["B", "C"], f"beam search should keep delayed best path: {beam}")
    print("PASS  beam search")


def test_normalization():
    raw = normalized_score(-4.0, length=4, alpha=0.0)
    norm = normalized_score(-4.0, length=4, alpha=1.0)
    check(raw == -4.0, f"alpha 0 should leave score unchanged: {raw}")
    check(norm == -1.0, f"alpha 1 should divide by length: {norm}")
    print("PASS  normalization")


if __name__ == "__main__":
    test_beam_search()
    test_normalization()
    print("All tests passed.")
