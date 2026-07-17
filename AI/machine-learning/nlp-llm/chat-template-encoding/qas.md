# Chat Template Encoding - Q&A

---

## Intuition

**Q: What does a chat template do?**
A: It converts a list of role/content messages into the string format expected by a chat model.

**Q: Is chat templating part of the tokenizer or the model weights?**
A: It is tokenizer-side formatting.

**Q: Why do models need different chat templates?**
A: They were instruction-tuned with different control tokens and conversation formats.

---

## 1. Messages

**Q: What keys does a standard chat message contain?**
A: `role` and `content`.

**Q: What are common roles?**
A: `system`, `user`, and `assistant`.

**Q: What happens to the message list before generation?**
A: It is rendered into one tokenizable string.

---

## 2. Generation Prompt

**Q: What does `add_generation_prompt=True` add?**
A: The assistant-start marker where the model should begin answering.

**Q: Should the assistant generation prompt include answer content?**
A: No. It marks where new assistant content should be generated.

**Q: Why inspect `tokenize=False` output?**
A: To debug the exact rendered prompt format.

---

## 3. Gotchas

**Q: Can the same messages produce different strings across tokenizers?**
A: Yes. Each tokenizer can define its own template.

**Q: What is a common manual formatting mistake?**
A: Adding duplicate control tokens that the template already inserts.

**Q: What does the model actually receive?**
A: Token IDs produced from the rendered chat string.

---

## 4. Practical Use

**Q: What Hugging Face method applies a chat template?**
A: `tokenizer.apply_chat_template(...)`.

**Q: What argument returns a string instead of token IDs?**
A: `tokenize=False`.

**Q: What argument returns tensors for generation?**
A: `return_tensors`, commonly with `tokenize=True`.
