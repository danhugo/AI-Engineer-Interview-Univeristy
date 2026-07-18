# Bayesian Hyperparameter Optimization - Q&A

---

## Core Idea

**Q: What is the surrogate model?**
A: A model that predicts validation score from hyperparameters.

**Q: What does an acquisition function do?**
A: Chooses the next hyperparameter trial.

**Q: What tradeoff does it manage?**
A: Exploration versus exploitation.

## Usage

**Q: When is Bayesian optimization useful?**
A: When trials are expensive and budget is limited.

**Q: When might random search be enough?**
A: When trials are cheap or the space is simple.
