# Break-Even Pay-Per-Token API vs Dedicated GPU - Interview Knowledge Sheet

## Intuition

Pay-per-token APIs charge for usage.

Dedicated GPUs charge for time, whether or not the GPU is fully used.

Break-even asks:

$$
\text{API token cost} = \text{GPU hourly cost}
$$

The decision is not only price. APIs also include model quality, operations, scaling, uptime, safety tooling, and model updates. Dedicated GPUs give control and can be cheaper at high sustained utilization.

---

## 1. API Cost

If prices are per million tokens:

$$
\text{API cost} =
\frac{I}{10^6}P_i +
\frac{O}{10^6}P_o
$$

where:

- `I`: input tokens
- `O`: output tokens
- `P_i`: dollars per million input tokens
- `P_o`: dollars per million output tokens

Cached input, batch discounts, priority processing, and long-context multipliers can change this formula.

---

## 2. GPU Cost

For a dedicated serving stack:

$$
\text{GPU cost per hour} =
\text{GPU rental} + \text{CPU} + \text{memory} + \text{storage} + \text{network} + \text{ops}
$$

If the GPU produces `R` billable tokens per second at utilization `u`, then hourly tokens are:

$$
3600 \cdot R \cdot u
$$

Cost per million tokens is:

$$
\frac{\text{hourly cost}}{3600 \cdot R \cdot u} \cdot 10^6
$$

---

## 3. Break-Even Throughput

If an API workload has blended cost `C_api` dollars per million tokens and a GPU costs `C_gpu` dollars per hour, the break-even generated billable token rate is:

$$
R_{\text{break-even}} =
\frac{C_{\text{gpu}} \cdot 10^6}{3600 \cdot C_{\text{api}}}
$$

Above that sustained rate, dedicated serving may be cheaper on raw compute.

Below that rate, pay-per-token often wins because idle time is not billed.

---

## 4. What To Include In Interviews

Include:

- input/output token mix
- cache hit rate
- target latency SLO
- sustained utilization
- batchability
- engineering and on-call cost
- model quality and feature gap
- burstiness and idle time

The biggest mistake is comparing API list price against GPU rental price without utilization.

---

## Current Price Examples

As of July 17, 2026, public pages list examples such as OpenAI model prices per 1M tokens and cloud GPU prices per second or hour. These are examples for modeling, not constants.

Always re-check prices before a real purchasing decision.

---

## References

- OpenAI model pricing in API docs: https://developers.openai.com/api/docs/models
- OpenAI token counting help: https://help.openai.com/en-us/articles/4936856-how-can-i-estimate-the-cost-of-my-usage
- Modal pricing: https://modal.com/pricing
- Runpod GPU pricing: https://www.runpod.io/pricing
