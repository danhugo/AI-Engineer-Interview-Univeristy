"""Reference solutions for Chat Template Encoding."""


ROLE_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def render_chat(messages, add_generation_prompt=False):
    """
    Render messages into a simple chat-template string.

    Format each message as:
      <|role|>
      content</s>
    """
    pieces = []
    for message in messages:
        role = message["role"]
        if role not in ROLE_MARKERS:
            raise ValueError(f"unsupported role: {role}")
        pieces.append(f"{ROLE_MARKERS[role]}\n{message['content']}</s>\n")
    if add_generation_prompt:
        pieces.append(f"{ROLE_MARKERS['assistant']}\n")
    return "".join(pieces)


def whitespace_tokenize(text, vocab):
    """Tokenize by whitespace and map tokens to IDs."""
    ids = []
    for token in text.split():
        if token not in vocab:
            raise ValueError(f"unknown token: {token}")
        ids.append(vocab[token])
    return ids


def apply_chat_template(messages, tokenize=False, add_generation_prompt=False, vocab=None):
    """Render chat messages, optionally returning token IDs."""
    rendered = render_chat(messages, add_generation_prompt=add_generation_prompt)
    if not tokenize:
        return rendered
    if vocab is None:
        raise ValueError("vocab is required when tokenize=True")
    return whitespace_tokenize(rendered, vocab)


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def test_render_chat():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Define BPE."},
    ]
    rendered = render_chat(messages, add_generation_prompt=True)
    expected = (
        "<|system|>\nBe concise.</s>\n"
        "<|user|>\nDefine BPE.</s>\n"
        "<|assistant|>\n"
    )
    check(rendered == expected, f"rendered template wrong: {rendered!r}")
    print("PASS  render")


def test_tokenize_and_validate():
    messages = [{"role": "user", "content": "Hi"}]
    vocab = {"<|user|>": 0, "Hi</s>": 1, "<|assistant|>": 2}
    ids = apply_chat_template(messages, tokenize=True, add_generation_prompt=True, vocab=vocab)
    check(ids == [0, 1, 2], f"token IDs wrong: {ids}")

    try:
        render_chat([{"role": "developer", "content": "Nope"}])
    except ValueError:
        print("PASS  validation")
        return
    raise AssertionError("FAIL  unknown role should raise ValueError")


if __name__ == "__main__":
    test_render_chat()
    test_tokenize_and_validate()
    print("All tests passed.")
