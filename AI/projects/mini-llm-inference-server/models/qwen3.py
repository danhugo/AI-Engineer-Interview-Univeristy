"""Stage-1 Qwen3 forward pass.

Deliberately the SIMPLEST version that is still correct:
  - single GPU, no tensor parallelism
  - no paged KV cache (we recompute the whole sequence each step)
  - flash-attention for the attention math

Module + parameter names match HuggingFace Qwen3 exactly, so loading weights is
just `model.load_state_dict(hf_model.state_dict())` — no name remapping.

Paged attention and tensor parallelism replace pieces of this file in later stages.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

# flash-attn needs a CUDA GPU and fp16/bf16 tensors. Import is optional so this
# file still runs on a Mac or a pre-Ampere GPU via the SDPA fallback.
try:
    from flash_attn import flash_attn_func
except ImportError:  # pragma: no cover - depends on the machine
    flash_attn_func = None

from layers.attention import paged_attend
from utils.context import get_context


def attend(q, k, v):
    """Causal grouped-query attention.

    Uses flash-attn when it can (fp16/bf16 on CUDA), else torch SDPA. SDPA is
    needed for fp32 — flash-attn rejects it — which is how we verify that a
    bf16 mismatch is precision and not a bug.
    """
    if flash_attn_func is not None and q.dtype in (torch.float16, torch.bfloat16):
        # flash-attn takes (batch, seq, heads, dim) and handles GQA natively.
        return flash_attn_func(q, k, v, causal=True)

    # SDPA wants (batch, heads, seq, dim); enable_gqa broadcasts the kv heads.
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return o.transpose(1, 2)


def rope_theta(config: Qwen3Config) -> float:
    """Read the RoPE base.

    transformers 5.x moved it from `config.rope_theta` into the
    `config.rope_parameters` dict. Support both. Qwen3 uses 1e6, not the 1e4
    default, so getting this wrong silently wrecks long-context positions.
    """
    params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return params["rope_theta"]
    return getattr(config, "rope_theta", 1_000_000)


def head_dim_of(config: Qwen3Config) -> int:
    """Qwen3 sets head_dim explicitly — it is not always hidden_size // num_heads."""
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. Qwen normalizes in float32, then casts back."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and rotate — the HF RoPE convention."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to q and k.

    q, k: (batch, seq, heads, head_dim)
    cos, sin: (seq, head_dim // 2), float32
    """
    # Duplicate each freq so it lines up with the full head_dim, then broadcast
    # over batch and heads: (1, seq, 1, head_dim).
    cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
    sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
    # Cast to q's dtype FIRST. cos/sin are float32 for precision, and bf16 * fp32
    # promotes the whole product to fp32 — which flash-attn rejects.
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class RotaryEmbedding(nn.Module):
    """Precompute inverse frequencies; produce cos/sin for given positions."""

    def __init__(self, head_dim: int, base: float, max_position: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor):
        # positions: (seq,) -> freqs: (seq, head_dim // 2)
        # .float() on inv_freq too: model.to(bf16) converts float buffers, and we
        # want the angles computed in fp32 regardless of the model's dtype.
        freqs = torch.outer(positions.float(), self.inv_freq.float())
        return freqs.cos(), freqs.sin()


class Qwen3Attention(nn.Module):
    """Grouped-query attention with Qwen3's per-head QK-Norm."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim_of(config)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        bias = config.attention_bias  # Qwen3 = False
        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=bias)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)

        # QK-Norm: Qwen3 RMS-norms each head's q and k vectors. Always present.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Set by attach_kv_cache(). None means the stage-1 uncached path.
        self.k_cache = None
        self.v_cache = None

    def forward(self, hidden_states, cos, sin):
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim)

        # QK-Norm is applied per head, BEFORE RoPE (matches HF ordering).
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        # Both backends default the softmax scale to 1/sqrt(head_dim), as we want.
        ctx = get_context()
        if self.k_cache is not None and ctx.slot_mapping is not None:
            o = paged_attend(q, k, v, self.k_cache, self.v_cache, ctx)
        else:
            o = attend(q, k, v)  # stage-1 path: no cache, recompute every step
        return self.o_proj(o.reshape(b, s, -1))


class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """Pre-norm transformer block: norm -> attn -> add, norm -> mlp -> add."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, cos, sin)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen3Model(nn.Module):

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(
            head_dim_of(config),
            base=rope_theta(config),
            max_position=config.max_position_embeddings,
        )

    def forward(self, input_ids, positions):
        # input_ids: (batch, seq)   positions: (seq,)
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Small Qwen3 models share input/output embeddings.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions):
        hidden_states = self.model(input_ids, positions)
        return self.lm_head(hidden_states)  # logits: (batch, seq, vocab)
