# Expert Parallelism Token Routing Communication Cost - Interview Knowledge Sheet

## Intuition

Mixture-of-Experts models activate only a few feed-forward experts per token.

Expert parallelism places different experts on different GPUs. A router chooses top-k experts for each token, then the system sends token activations to the GPUs that own those experts and gathers the outputs.

The key systems cost is communication:

$$
\text{tokens} \rightarrow \text{all-to-all dispatch} \rightarrow \text{local experts} \rightarrow \text{all-to-all combine}
$$

---

## 1. Routing Shape

For `T` tokens and top-k routing, the router produces:

$$
\text{routes} \in \mathbb{N}^{T \times k}
$$

Each token creates `k` expert assignments.

If there are `E` experts over `G` GPUs and experts are evenly sharded, each GPU owns:

$$
E / G
$$

experts.

---

## 2. Communication Estimate

Let:

- `T`: local tokens on a GPU
- `k`: experts per token
- `H`: hidden size
- `b`: bytes per activation element
- `r`: fraction of assignments that go to remote GPUs

Dispatch traffic per GPU is approximately:

$$
T \cdot k \cdot H \cdot b \cdot r
$$

The combine path returns expert outputs of similar size:

$$
2 \cdot T \cdot k \cdot H \cdot b \cdot r
$$

This is a useful first-order estimate. Real systems also pay for metadata, padding, capacity buffers, synchronization, and load imbalance.

---

## 3. Why Load Balance Matters

If many tokens choose the same expert, that expert's GPU becomes a bottleneck.

MoE systems use techniques such as auxiliary load-balancing losses, capacity factors, token dropping, or grouped routing to avoid one expert receiving too many tokens.

The routing quality affects both model quality and hardware efficiency.

---

## 4. Top-1 vs Top-2

Top-2 routing can improve model quality because each token sees two experts.

But it roughly doubles expert compute and dispatched activation traffic:

$$
\text{assignments} = T \cdot k
$$

So increasing `k` is expensive at serving time.

---

## Interview Gotchas

- Expert parallelism shards experts, not individual dense layers.
- Routing creates all-to-all communication when selected experts are remote.
- Top-k routing multiplies token assignments.
- Communication cost depends on hidden size and dtype, not vocabulary size.
- Load imbalance can dominate the average-case cost estimate.

---

## References

- DeepSpeed MoE layer API: https://deepspeed.readthedocs.io/en/stable/moe.html
- DeepSpeed MoE inference tutorial: https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/
- Hugging Face Transformers expert parallelism docs: https://huggingface.co/docs/transformers/expert_parallelism
- Triton-distributed all-to-all expert parallel tutorial: https://triton-distributed.readthedocs.io/en/latest/getting-started/tutorials/04-deepseek-infer-all2all.html
