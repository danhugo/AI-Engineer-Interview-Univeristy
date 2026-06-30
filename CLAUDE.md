# AI Engineer Interview University — CLAUDE.md

> **Most important rule: be concise, use simple words. No fancy academic language.**

## What This Is

A self-study course covering DSA and AI engineering topics for interview preparation. Each topic lives in its own folder under `DSA/` (and future `AI/` or `Systems/` folders).

## Folder Structure per Topic

```
DSA/<topic>/
  note.md       ← concept notes (the source of truth)
  qas.md        ← Q&A pairs derived from note.md
  practice.py   ← coding practice stubs
  solution.py   ← reference solutions
```

## Content Design Pattern

### note.md
- Starts with a plain-English analogy or one-line summary
- Covers: concept → internals → operations → complexity → common interview patterns
- Language: simple and direct — no academic jargon unless necessary
- Code examples for anything non-obvious
- Complexity tables for operations

### qas.md
- Every section in note.md has corresponding Q&A pairs
- Format: `**Q: question?**` / `A: answer`
- Answers are 1–4 sentences — enough to say it clearly, no padding
- Mirrors note.md structure and section order exactly
- No content that isn't in note.md

### practice.py
- Header docstring explains the file's purpose and a numbered HOW TO USE guide
- Student replaces `pass` with their own code
- Each method has a short docstring: one sentence of what it does, then a `HINT:` block showing the approach (not the answer)
- Hints name the specific steps and data structures to use — enough to guide, not enough to copy
- Built-in test suite at the bottom: run `python practice.py` to see PASS/FAIL per method

### solution.py
- Identical class/method signatures as practice.py — drop-in replacement
- Clean, commented implementations — comments explain the *why* of non-obvious lines
- No extra features beyond what practice.py asks for
- Serves as the reference after the student has attempted the practice

### Writing rules (apply to all files)

**Most important: be concise, use simple words. No fancy academic language.**

- Short sentences. If a sentence needs a comma, split it.
- Explain the *why*, not just the *what*
- Use concrete examples over abstract descriptions ("two memory jumps" not "indirection overhead")
- No filler phrases, no padding
- When rewriting: simplify first, add content only if there's a clear gap

## Topics Covered So Far

| Topic | Folder |
|-------|--------|
| Array | `DSA/array/` |
| Hash Table | `DSA/hash-table/` |
