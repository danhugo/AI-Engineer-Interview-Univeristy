import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.qwen3 import Qwen3ForCausalLM

# Resolves from the HF cache on the GPU box (sync.sh sets HF_HOME=~/mini-llm/hf).
path = "Qwen/Qwen3-8B"
PROMPT = "The specialty of Hanoi is"
MAX_NEW_TOKENS = 40

# Check correctness at several lengths. A 6-token prompt barely exercises RoPE;
# bugs in rotary angles, GQA head mapping, or the causal mask usually only show
# up at high position indices.
CHECK_LENGTHS = (6, 128, 512, 2048)

# Varied prose, tiled up to the longest length we check. Content repeats but
# position indices do not, so RoPE is still tested across the whole range.
FILLER = """
Hanoi sits on the western bank of the Red River, a city of lakes and narrow
streets where the old quarter still follows the outlines of craft guilds that
settled there centuries ago. Machine learning systems, by contrast, are built
from parts that did not exist a decade ago: attention layers that compare every
token against every other, key-value caches that trade memory for speed, and
schedulers that decide which request advances next. A transformer reads a
sequence, projects each token into queries, keys, and values, and mixes them
according to how strongly they attend to one another. The cost grows with the
square of the sequence length, which is why caching matters so much in
practice. Rotary embeddings encode position by rotating pairs of dimensions at
frequencies that fall off geometrically, so nearby tokens differ by small
angles and distant ones by large ones. Grouped-query attention shares a single
key-value head across several query heads, cutting the cache in half or better
without much loss in quality. Paged memory borrows an idea from operating
systems: split the cache into fixed blocks, keep a table mapping logical
positions to physical blocks, and fragmentation stops being a problem.
"""


def load_models(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Same attention backend as ours (flash-attn), so any logit difference comes
    # from our code, not from SDPA-vs-flash-attn numerical noise.
    hf = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    hf = hf.cuda().eval()

    ours = Qwen3ForCausalLM(hf.config).to(torch.bfloat16)
    ours.load_state_dict(hf.state_dict())  # both sides bf16 now — no upcast round-trip
    ours = ours.cuda().eval()

    return tokenizer, hf, ours


@torch.inference_mode()
def check_correctness(
    hf, ours, ids: torch.Tensor, top_k: int = 5, min_agreement: float = 0.95
) -> None:
    """Compare logits against HF across every position.

    vLLM's own check_logprobs_close (tests/models/utils.py), a position "matches" if each
    model's top-1 pick is at least in the OTHER model's top-k.
    """
    positions = torch.arange(ids.shape[1], device=ids.device)

    ref_logits = hf(ids).logits[0]        # (seq, vocab)
    our_logits = ours(ids, positions)[0]  # (seq, vocab)

    diff = (our_logits.float() - ref_logits.float()).abs().max().item()

    # Absolute logit diff is misleading on its own: tiled text makes the model
    # very confident, logits grow, and the same relative error looks huge. What
    # matters for inference is the probability distribution, so report that too.
    scale = ref_logits.float().abs().max().item()
    ref_p = ref_logits.float().softmax(-1)
    our_p = our_logits.float().softmax(-1)
    max_prob_diff = (ref_p - our_p).abs().max().item()
    # Total variation distance, worst position: half the L1 gap between dists.
    max_tv = (ref_p - our_p).abs().sum(-1).max().item() / 2
    del ref_p, our_p

    ref_next = ref_logits.argmax(-1)             # (seq,)
    our_next = our_logits.argmax(-1)              # (seq,)
    ref_topk = ref_logits.topk(top_k, dim=-1).indices  # (seq, top_k)
    our_topk = our_logits.topk(top_k, dim=-1).indices  # (seq, top_k)

    exact_match = ref_next == our_next
    topk_match = (our_next.unsqueeze(-1) == ref_topk).any(-1) & (
        ref_next.unsqueeze(-1) == our_topk
    ).any(-1)
    agreement = topk_match.float().mean().item()

    print(f"[correctness] max abs logit diff: {diff:.4f}  (logit scale {scale:.1f}, relative {diff / scale:.2%})")
    print(f"[correctness] max prob diff: {max_prob_diff:.4f}   worst-position TV distance: {max_tv:.4f}")
    print(f"[correctness] exact top-1 match: {exact_match.float().mean().item():.1%}")
    bad = (~exact_match).nonzero().flatten().tolist()
    if bad:
        print(f"[correctness] top-1 mismatch at positions: {bad[:20]}{' ...' if len(bad) > 20 else ''}")
    print(f"[correctness] top-{top_k} mutual agreement over {ids.shape[1]} positions: {agreement:.1%}")
    assert agreement >= min_agreement, (
        f"top-{top_k} agreement {agreement:.1%} below {min_agreement:.0%} — "
        "reimplementation likely has a bug, not just numerical noise"
    )
    print("[correctness] OK\n")


@torch.inference_mode()
def generate(ours, tokenizer, ids: torch.Tensor, max_new_tokens: int) -> str:
    """Greedy decode, no KV cache — recompute the whole sequence every step."""
    for _ in range(max_new_tokens):
        positions = torch.arange(ids.shape[1], device=ids.device)
        logits = ours(ids, positions)
        next_id = logits[:, -1, :].argmax(-1)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def build_long_ids(tokenizer, n: int) -> torch.Tensor:
    """Tokenize FILLER and tile it to exactly n tokens."""
    base = tokenizer(FILLER, return_tensors="pt").input_ids[0]
    reps = -(-n // len(base))  # ceil division
    return base.repeat(reps)[:n].unsqueeze(0).cuda()


def run():
    tokenizer, hf, ours = load_models(path)

    for n in CHECK_LENGTHS:
        ids = build_long_ids(tokenizer, n)
        print(f"--- seq_len {n} ---")
        check_correctness(hf, ours, ids)
        del ids
        torch.cuda.empty_cache()

    del hf
    torch.cuda.empty_cache()

    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()

    text = generate(ours, tokenizer, ids, MAX_NEW_TOKENS)
    print(f"[generate] {text}")


if __name__ == "__main__":
    run()
