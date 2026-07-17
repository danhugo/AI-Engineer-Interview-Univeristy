# Rotary Positional Encoding RoPE - Q&A

---

## Intuition

**Q: What does RoPE do?**
A: It encodes positions by rotating query and key vectors according to token position.

**Q: Why is it called rotary?**
A: Each pair of hidden dimensions is treated like a 2D vector and rotated.

**Q: What problem does RoPE solve?**
A: It gives attention position information without relying on a learned absolute position table.

---

## Mechanics

**Q: Which tensors usually receive RoPE?**
A: Queries and keys.

**Q: Why not values?**
A: Position should affect attention weights through query-key scores; values are the content mixed by those weights.

**Q: What does `rotate_half([a, b])` return for one pair?**
A: `[-b, a]`.

**Q: Why must the head dimension be even?**
A: RoPE rotates dimensions in pairs.

---

## Properties

**Q: Does RoPE preserve vector norms?**
A: Yes, rotation preserves length.

**Q: How does RoPE create relative-position behavior?**
A: The dot product of rotated query and key vectors depends on the difference between their positions.

**Q: Does RoPE use learned parameters?**
A: The basic version does not; it uses fixed frequencies.

---

## Interview Gotchas

**Q: Is RoPE the same as sinusoidal embeddings added to token embeddings?**
A: No. It uses sinusoidal frequencies to rotate queries and keys.

**Q: Is RoPE absolute or relative?**
A: It applies absolute-position rotations that make attention scores encode relative offsets.
