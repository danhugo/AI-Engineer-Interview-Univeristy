# Adam / AdamW Optimizer Step — Q&A

---

## Core Idea

**Q: What does Adam add beyond plain SGD?**
A: Moving averages of gradients and squared gradients.

**Q: What does Adam's first moment track?**
A: The running average direction of the gradients.

**Q: What does Adam's second moment track?**
A: The running average of squared gradients.

**Q: Why is Adam called adaptive?**
A: Each parameter gets scaled by its own historical gradient size.

---

## Formula

**Q: What is the first-moment update?**
A: `m = beta1 * m + (1 - beta1) * grad`.

**Q: What is the second-moment update?**
A: `v = beta2 * v + (1 - beta2) * grad**2`.

**Q: Why use bias correction?**
A: `m` and `v` start at zero, so early moving averages are biased low.

**Q: What is the Adam parameter update?**
A: `theta = theta - lr * m_hat / (sqrt(v_hat) + eps)`.

---

## AdamW

**Q: What does AdamW change?**
A: It decouples weight decay from the Adam gradient update.

**Q: Why does decoupling matter?**
A: In adaptive optimizers, adding L2 into the gradient is not the same as direct weight decay.

**Q: What is the direct weight decay part?**
A: `theta = theta - lr * weight_decay * theta`.

**Q: When should you prefer AdamW over Adam?**
A: When using Adam-style optimization with weight decay.

---

## PyTorch

**Q: What optimizer class implements AdamW in PyTorch?**
A: `torch.optim.AdamW`.

**Q: What is the usual training order?**
A: `zero_grad`, forward pass, loss, `backward`, `step`.

**Q: What does `eps` do?**
A: It prevents division by zero and improves numerical stability.
