"""Reference solutions for Extend BPE Tokenizer."""


def add_tokens(vocab, tokens):
    """
    Return a copied vocab with new normal tokens appended.

    Existing tokens should not receive new IDs.
    """
    new_vocab = dict(vocab)
    next_id = max(new_vocab.values(), default=-1) + 1
    for token in tokens:
        if token not in new_vocab:
            new_vocab[token] = next_id
            next_id += 1
    return new_vocab


def add_special_tokens(vocab, special_tokens):
    """
    Return (new_vocab, special_token_set).

    Special tokens should also be present in the vocabulary.
    """
    new_vocab = add_tokens(vocab, special_tokens)
    return new_vocab, set(special_tokens)


def resize_embedding_shape(old_shape, new_vocab_size):
    """Return the new embedding matrix shape after extending vocabulary."""
    _, hidden_dim = old_shape
    return (new_vocab_size, hidden_dim)


def encode_longest_match(text, vocab):
    """
    Encode text using greedy longest-token matching.

    This is a tiny stand-in for testing added-token atomic behavior.
    """
    tokens_by_length = sorted(vocab, key=len, reverse=True)
    ids = []
    i = 0
    while i < len(text):
        match = None
        for token in tokens_by_length:
            if text.startswith(token, i):
                match = token
                break
        if match is None:
            raise ValueError(f"no token can encode text at offset {i}: {text[i:]}")
        ids.append(vocab[match])
        i += len(match)
    return ids


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_add_tokens_and_resize():
    vocab = {"l": 0, "o": 1, "w": 2, "low": 3}
    extended = add_tokens(vocab, ["lower", "low"])
    check(extended["lower"] == 4, f"new token should be appended: {extended}")
    check(extended["low"] == 3, "existing token ID should not change")
    check(resize_embedding_shape((4, 8), len(extended)) == (5, 8), "embedding rows should match vocab")
    print("PASS  add tokens")


def test_special_token_atomic_encoding():
    vocab = {"h": 0, "i": 1, " ": 2}
    extended, specials = add_special_tokens(vocab, ["<|assistant|>"])
    ids = encode_longest_match("<|assistant|> hi", extended)
    check(specials == {"<|assistant|>"}, f"wrong special set: {specials}")
    check(ids[0] == extended["<|assistant|>"], f"special token should be atomic: {ids}")
    check(ids[-2:] == [extended["h"], extended["i"]], f"characters should still encode: {ids}")
    print("PASS  special tokens")


if __name__ == "__main__":
    test_add_tokens_and_resize()
    test_special_token_atomic_encoding()
    print("All tests passed.")
