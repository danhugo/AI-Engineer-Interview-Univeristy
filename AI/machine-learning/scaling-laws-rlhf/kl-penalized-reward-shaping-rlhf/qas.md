# KL-Penalized Reward Shaping RLHF - Q&A

---

## Core Idea

**Q: Why add a KL penalty?**
A: To keep the policy close to the reference and reduce reward hacking.

**Q: What does beta control?**
A: Strength of the KL penalty.

**Q: What happens when beta is too small?**
A: The policy can drift and exploit the reward model.
