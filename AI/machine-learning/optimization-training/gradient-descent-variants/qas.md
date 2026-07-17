# Gradient Descent Variants — Q&A

---

## Core Idea

**Q: What does gradient descent do?**
A: It repeatedly updates parameters in the direction that lowers the loss.

**Q: What is the basic update rule?**
A: `theta = theta - learning_rate * gradient`.

**Q: What does the gradient point toward?**
A: The direction of steepest increase in the loss.

**Q: Why do we subtract the gradient?**
A: Subtracting moves parameters toward lower loss.

---

## Batch Gradient Descent

**Q: What data does batch gradient descent use per update?**
A: The full training dataset.

**Q: What is the main benefit of batch gradient descent?**
A: The gradient estimate is stable and deterministic.

**Q: What is the main downside?**
A: Each update can be expensive on large datasets.

---

## Stochastic Gradient Descent

**Q: What data does stochastic gradient descent use per update?**
A: One training example.

**Q: Why is SGD noisy?**
A: One example may not represent the full dataset gradient.

**Q: Why can SGD still work well?**
A: It makes cheap frequent updates, and the noise can help exploration.

---

## Mini-Batch Gradient Descent

**Q: What data does mini-batch gradient descent use per update?**
A: A small group of examples.

**Q: Why is mini-batch training common in deep learning?**
A: It balances noisy-but-cheap updates with efficient vectorized hardware use.

**Q: In practice, what does "SGD" often mean in deep learning code?**
A: Mini-batch SGD.

---

## Epochs and Shuffling

**Q: What is an epoch?**
A: One full pass through the training dataset.

**Q: Why shuffle each epoch?**
A: To avoid repeatedly training on examples in the same order.

---

## Learning Rate

**Q: What happens if the learning rate is too small?**
A: Training can be very slow.

**Q: What happens if the learning rate is too large?**
A: The loss can bounce, overshoot, or diverge.
