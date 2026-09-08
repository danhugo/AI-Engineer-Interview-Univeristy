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

from layers.attention import attend
from utils.context import get_context


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

    q, k: (num_tokens, heads, head_dim)
    cos, sin: (num_tokens, head_dim // 2), float32
    """
    # Duplicate each freq to cover the full head_dim, then add a head axis so it
    # broadcasts over heads: (num_tokens, 1, head_dim).
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)
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
        # hidden_states: (num_tokens, hidden_size)
        t = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(t, self.num_kv_heads, self.head_dim)

        # QK-Norm is applied per head, BEFORE RoPE (matches HF ordering).
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        # Both backends default the softmax scale to 1/sqrt(head_dim), as we want.
        o = attend(q, k, v, self.k_cache, self.v_cache, get_context())
        return self.o_proj(o.reshape(t, -1))


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
        # input_ids, positions: (num_tokens,)
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

    def forward(self, input_ids, positions, logits_indices=None):
        """input_ids, positions: (num_tokens,). Returns (num_tokens, vocab).

        A leading batch dim of 1 is accepted and restored, so the stage-1
        scripts keep working. Real batches go through the flat layout with
        cu_seqlens in the context — a batch dim cannot express sequences of
        different lengths without padding.

        logits_indices picks which rows to run the lm_head on. During prefill
        only the last token of each sequence is needed, and the vocab is 152k
        wide, so computing the rest is pure waste.
        """
        batched = input_ids.dim() == 2
        if batched:
            assert input_ids.shape[0] == 1, \
                "2D input is only for single-sequence compatibility; " \
                "batches must use the flat layout"
            input_ids = input_ids.reshape(-1)
        if positions.dim() == 2:
            positions = positions.reshape(-1)

        hidden_states = self.model(input_ids, positions)
        if logits_indices is not None:
            hidden_states = hidden_states[logits_indices]
        logits = self.lm_head(hidden_states)
        return logits.unsqueeze(0) if batched else logits
