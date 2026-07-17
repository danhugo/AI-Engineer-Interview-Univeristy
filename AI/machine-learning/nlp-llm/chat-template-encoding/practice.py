"""AI / Machine Learning - Chat Template Encoding Practice."""


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
    pass


def whitespace_tokenize(text, vocab):
    """Tokenize by whitespace and map tokens to IDs."""
    pass


def apply_chat_template(messages, tokenize=False, add_generation_prompt=False, vocab=None):
    """Render chat messages, optionally returning token IDs."""
    pass


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
