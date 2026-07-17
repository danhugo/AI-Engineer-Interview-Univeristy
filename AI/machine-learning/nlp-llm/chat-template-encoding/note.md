# Chat Template Encoding - Interview Knowledge Sheet

## Intuition

Chat models are still next-token prediction models.

They do not receive a Python list of messages directly.

A chat template converts messages like:

```python
{"role": "user", "content": "Hello"}
```

into one model-specific string with role markers, separators, and generation prompts.

---

## 1. Why Templates Matter

Different instruction-tuned models are trained with different formats.

One model may expect:

```text
[INST] hello [/INST]
```

Another may expect:

```text
<|user|>
hello</s>
<|assistant|>
```

Using the wrong template is a prompt formatting bug, not a model architecture issue.

---

## 2. Message Schema

The common schema is a list of dictionaries:

```python
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain beam search."},
]
```

The template renders roles and content into the exact text the model was trained to continue.

---

## 3. Generation Prompt

For generation, the input should usually end where the assistant is expected to begin.

That is the purpose of `add_generation_prompt=True`.

It appends the assistant prefix without assistant content:

```text
<|assistant|>
```

The model then continues from that point.

---

## 4. Tokenize vs Render

Chat templating can either:

- render a string for inspection
- render and tokenize for model input IDs

In Hugging Face:

```python
tokenizer.apply_chat_template(messages, tokenize=False)
tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
```

Inspecting the rendered string is often the fastest way to debug formatting mistakes.

---

## 5. Interview Gotchas

- Chat templates live on the tokenizer, not the model forward pass.
- The same message list can render differently for different models.
- `add_generation_prompt=True` prepares the assistant turn.
- Do not manually add duplicate BOS/EOS/control tokens if the template already does it.
- Role order and supported roles are model/template dependent.

---

## References

- Hugging Face chat templates guide: https://huggingface.co/docs/transformers/v4.43.0/chat_templating
- Hugging Face `apply_chat_template` API: https://huggingface.co/docs/transformers/main_classes/tokenizer
