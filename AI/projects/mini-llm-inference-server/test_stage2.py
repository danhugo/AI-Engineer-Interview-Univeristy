"""Stage 2 gate: paged KV cache must not change what the model generates.

Run: ./sync.sh py test_stage2.py

The cache is a pure optimisation — it stores K/V instead of recomputing it. So
cached output must match the stage-1 uncached output token for token. Any
difference is a bug in the block table, the slot mapping, or the positions.

Also checks a block boundary is crossed, since off-by-one errors there are the
most likely mistake and a short prompt would never reach it.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.generate import generate
from models.qwen3 import Qwen3ForCausalLM
from common import build_long_ids, path

MAX_NEW = 40
# flash-attn's paged kernel requires the block size to be a multiple of 256,
# which is why nano-vllm uses 256 too. So to exercise block boundaries we need
# a long prompt rather than small blocks.
BLOCK_SIZE = 256
PROMPT_LEN = 600  # spans 3 blocks once we generate


@torch.inference_mode()
def uncached_generate(model, ids, max_new_tokens, eos_token_id):
    """Stage-1 reference: recompute the whole sequence every step."""
    for _ in range(max_new_tokens):
        positions = torch.arange(ids.shape[1], device=ids.device)
        next_id = model(ids, positions)[:, -1, :].argmax(-1)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        if next_id.item() == eos_token_id:
            break
    return ids[0].tolist()


def main():
    tokenizer = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = Qwen3ForCausalLM(hf.config).to(torch.bfloat16)
    model.load_state_dict(hf.state_dict())
    model = model.cuda().eval()
    del hf
    torch.cuda.empty_cache()

    prompt_ids = build_long_ids(tokenizer, PROMPT_LEN)
    eos = tokenizer.eos_token_id
    print(f"[prompt] {prompt_ids.shape[1]} tokens")

    # 1. Reference, no cache at all.
    want = uncached_generate(model, prompt_ids, MAX_NEW, eos)
    print(f"[uncached] {len(want)} tokens")

    # 2. Same thing through the paged cache.
    manager = build_manager_for(model, num_blocks=64, block_size=BLOCK_SIZE)
    print(f"[cache] {manager.num_blocks} blocks x {BLOCK_SIZE} tokens = "
          f"{manager.capacity_tokens()} tokens, "
          f"{manager.bytes_per_token() / 1024:.0f} KB/token")

    got = generate(model, manager, prompt_ids[0].tolist(), MAX_NEW, eos)
    print(f"[cached]   {len(got)} tokens")

    blocks_used = -(-len(got) // BLOCK_SIZE)
    print(f"[cache] spanned {blocks_used} blocks (boundary crossing exercised)")
    assert blocks_used >= 3, "prompt too short to test block boundaries"
    assert manager.num_free_blocks == manager.num_blocks, "blocks leaked"

    print(f"\n[text] {tokenizer.decode(got, skip_special_tokens=True)}")

    if got == want:
        print("\n[stage2] PASS — cached output identical to uncached")
        return

    # Report the first divergence rather than a bare assert.
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    print(f"\n[stage2] FAIL — diverges at token {first} (prompt is {len(prompt_ids[0])} tokens)")
    print(f"  uncached: {want[max(0, first - 3):first + 3]}")
    print(f"  cached:   {got[max(0, first - 3):first + 3]}")
    print(f"  uncached text: {tokenizer.decode(want, skip_special_tokens=True)!r}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
