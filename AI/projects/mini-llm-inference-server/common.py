"""Shared fixtures: the model path, loading, and test prompts.

Split out of the old run.py so the tests import data and loading from here
rather than from each other.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.qwen3 import Qwen3ForCausalLM

path = "Qwen/Qwen3-8B"
PROMPT = "The specialty of Hanoi is"

# Varied prose, tiled to whatever length a test needs. Content repeats but
# position indices do not, so RoPE is exercised across the whole range.
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


def load_models(path: str = path, dtype=torch.bfloat16, attn="flash_attention_2",
                devices=("cuda", "cuda")):
    """Load HuggingFace's Qwen3 and ours from the same weights.

    Returns (tokenizer, hf, ours). Both sides get the same attention backend so
    any difference comes from our code, not from SDPA-vs-flash-attn noise.

    fp32 forces attn="sdpa": flash-attn only accepts fp16/bf16. Pass two
    devices to put one model per GPU, which fp32 needs (32 GB each).
    """
    tokenizer = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=attn)

    ours = Qwen3ForCausalLM(hf.config).to(dtype)
    ours.load_state_dict(hf.state_dict())  # both sides same dtype — no upcast round-trip

    hf = hf.to(devices[0]).eval()
    ours = ours.to(devices[1]).eval()
    torch.cuda.empty_cache()
    return tokenizer, hf, ours


def load_ours(path: str = path, dtype=torch.bfloat16, attn="flash_attention_2"):
    """Just our model, with HF's weights. HF is freed before returning."""
    tokenizer, hf, ours = load_models(path, dtype, attn)
    del hf
    torch.cuda.empty_cache()
    return tokenizer, ours


def build_long_ids(tokenizer, n: int, device="cuda") -> torch.Tensor:
    """Tokenize FILLER and tile it to exactly n tokens. Shape (1, n)."""
    base = tokenizer(FILLER, return_tensors="pt").input_ids[0]
    reps = -(-n // len(base))  # ceil
    return base.repeat(reps)[:n].unsqueeze(0).to(device)
