# Extend BPE Tokenizer - Interview Knowledge Sheet

## Intuition

A BPE tokenizer represents text with a fixed vocabulary of subword units.

Extending a tokenizer means adding new tokens after training.

This is common when a model needs new domain terms or control tokens such as `<|tool|>`, `<image>`, or `<|assistant|>`.

---

## 1. BPE Reminder

Byte Pair Encoding starts from small units and repeatedly merges frequent adjacent pairs.

Example:

```text
l o w
lo w
low
```

The final merge table defines how text is split into subword tokens.

---

## 2. Normal Tokens vs Special Tokens

Normal added tokens are vocabulary entries that should be matched as units.

Special tokens are control markers. They should not be split, normalized, or treated like ordinary text.

In Hugging Face Tokenizers:

```python
tokenizer.add_tokens(["acetylsalicylic"])
tokenizer.add_special_tokens(["<|assistant|>"])
```

`add_special_tokens` marks tokens so they can be preserved during encoding and optionally skipped during decoding.

---

## 3. Embedding Resize

Adding tokenizer entries changes the tokenizer vocabulary size.

For a neural model, the token embedding matrix must match the tokenizer size:

$$
E \in \mathbb{R}^{V \times d}
$$

If vocabulary size changes from `V` to `V + n`, the embedding matrix needs `n` new rows.

In Transformers this is usually:

```python
model.resize_token_embeddings(len(tokenizer))
```

New rows are initialized; they are not automatically trained to be meaningful.

---

## 4. When To Extend

Good reasons:

- new chat/control markers
- domain terms that are repeatedly fragmented
- multimodal placeholders
- tool-use delimiters

Weak reasons:

- adding many rare words without fine-tuning
- trying to fix poor model behavior only through tokenization
- changing the tokenizer without resizing model embeddings

---

## 5. Interview Gotchas

- Added tokens get new IDs, usually after the existing vocabulary.
- Special tokens should remain atomic.
- Tokenizer and model vocabulary sizes must stay aligned.
- Adding tokens does not teach the model their meaning by itself.
- BPE merge rules and added-token matching are related but not identical mechanisms.

---

## References

- Hugging Face Tokenizers `Tokenizer` API: https://huggingface.co/docs/tokenizers/api/tokenizer
- Hugging Face Transformers tokenizer API: https://huggingface.co/docs/transformers/main_classes/tokenizer
- Sennrich, Haddow, Birch, "Neural Machine Translation of Rare Words with Subword Units": https://aclanthology.org/P16-1162/
