# Shortcut Connection Gradient Effect - Q&A

---

## Intuition

**Q: What is a shortcut connection?**
A: A connection that skips one or more transformations and is added to the transformed output.

**Q: What is the basic residual block formula?**
A: `y = x + F(x)`.

**Q: What does the residual branch learn?**
A: A correction to the input representation.

---

## Gradients

**Q: What is `dy/dx` for `y = x + F(x)`?**
A: `I + dF/dx`.

**Q: Why is the identity term useful?**
A: It gives gradients a direct path backward.

**Q: Does a shortcut guarantee gradients never vanish?**
A: No. It helps optimization but is not a complete guarantee.

---

## Optimization

**Q: Why can residual learning be easier than plain mapping learning?**
A: If identity is close to useful, the block can learn a small residual update.

**Q: What happens if the residual branch learns zero?**
A: The block behaves like the identity function.

**Q: Why does this help very deep networks?**
A: Extra layers can be easier to optimize because they can preserve representations when needed.

---

## Transformers

**Q: Where are shortcuts used in transformers?**
A: Around attention and feed-forward sublayers.

**Q: Why must sublayer output shape match input shape?**
A: Addition requires compatible shapes.

**Q: What is a projection shortcut?**
A: A learned projection on the skip path used when dimensions do not match.

---

## Gotchas

**Q: Is a residual connection concatenation?**
A: No. It is usually addition.

**Q: Does the shortcut branch add parameters?**
A: Identity shortcuts do not; projection shortcuts do.

**Q: Is the residual stream the same as the MLP branch?**
A: No. The residual stream carries the running representation; the branch computes an update.
