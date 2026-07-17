# Expert Parallelism Token Routing Communication Cost - Q&A

---

## Intuition

**Q: What is expert parallelism?**
A: A strategy where different MoE experts live on different devices.

**Q: What chooses the experts for each token?**
A: A router or gating network.

**Q: Why does expert parallelism need communication?**
A: A token's selected expert may live on another GPU.

---

## Routing

**Q: What does top-k routing mean?**
A: Each token is sent to `k` selected experts.

**Q: How many expert assignments are created for `T` tokens?**
A: `T * k`.

**Q: Why is top-2 more expensive than top-1?**
A: It doubles assignments and usually doubles routed expert activation traffic.

---

## Communication

**Q: What tensor is communicated to experts?**
A: Token hidden-state activations.

**Q: What is a simple dispatch byte estimate?**
A: `tokens * top_k * hidden_size * bytes_per_element * remote_fraction`.

**Q: Why is there also combine traffic?**
A: Expert outputs must return to the original token positions.

---

## Load Balance

**Q: Why can a single expert bottleneck serving?**
A: Too many routed tokens can overload the GPU that owns that expert.

**Q: What is a capacity factor?**
A: A limit or buffer controlling how many tokens an expert can process.

**Q: What is the main interview caveat?**
A: Average byte estimates ignore skew, padding, synchronization, and network topology.
