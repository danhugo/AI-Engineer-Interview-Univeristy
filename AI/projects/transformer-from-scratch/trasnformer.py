from torch import nn
import torch
import math


class Embedding(nn.Module):
    """
    Convert token IDs to embeddings.

    (batch, seq_len) -> (batch, seq_len, d_model)

    Args:
        num_embeddings (int): size of the dictionary of embeddings (size of the vocabulary)
        embedding_dim (int): embedding size.
        padding_idx (int, optional): token ID used for padding sequences same length within a batch
            Example:
            It is a cat         -> ['It', 'is', 'a', 'cat', PAD_TOKEN]
            It is a yellow cat  -> ['It', 'is', 'a', 'yellow', 'cat']

            embedding of padding_idx does not contribute to gradient and is not updated during training.
            By default it is all zeros.
    
    Attributes:
        weight (torch.Tensor): learnable weights of the module of shape (num_embeddings, embedding_dim)
            initialized from normal distribution nn.init.normal_()
    """
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int | None = None,
        ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

            # clean out gradients at padding_idx    
            self.weight.register_hook(self._zero_padding_grad)

    def _zero_padding_grad(self, grad: torch.Tensor) -> torch.Tensor:
        # PyTorch recommends not modifying the incoming gradient in-place 
        # because the grad passed to your hook may be shared or used by other hooks
        grad = grad.clone()
        grad[self.padding_idx] = 0
        return grad

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Index rows by token ID: (batch, seq_len) -> (batch, seq_len, d_model)
        return self.weight[input]

class PositionalEncoding(nn.Module):
    """
    Add positional information, because attention does not know token order.

    Without this, the model sees: 
    
    *I go to school, school to go I* 
    
    as the same set of tokens.
    """
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)    # use math.log to avoid unneccesary torch tensor overhead
        ) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_seq_len, d_model) -> broadcasts over batch
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model). Slice pe to this seq_len and add.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Lets each token looks at other tokens and decide what information matters.

    **encoder self-attention**: each source token looks at all source tokens

    **decoder masked self-attention**: each target token looks only at previous target tokens

    Args:
        d_model (int): model dimension.
        num_heads (int): number of attention heads. d_model = head_dim * num_heads
        dropout (float): a dropout layer on attention weights. Default: 0.0.
        bias: add bias as parameter. Default: True.


    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = False,
        ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, seq_len, d_model]
        Output: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1,2)
    
    def merge_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, num_heads, seq_len, head_dim]
        Output: [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1,2).contiguous()
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, query_len, d_model)
            key: (batch, key_len, d_model). Defaults to query for self-attention.
            value: (batch, key_len, d_model). Defaults to key.
            attn_mask: mask broadcastable to (batch, num_heads, query_len, key_len).
        """
        key = query if key is None else key
        value = key if value is None else value
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)
        
        attn_scores = torch.matmul(q, k.transpose(-2,-1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # Apply attention_mask before softmax.
        # attn_mask = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

        # attn_scores has shape (batch, num_heads, num_queries, num_keys)
        # applying softmax upon last dimension 
        # represent probabilities of each query token over all key tokens

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # intuition for attention dropout: 
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = self.merge_head(context)
        output = self.out_proj(context)
        return output

class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class EncoderBlock(nn.Module):
    """Convert source tokens to context-aware representations (Contextualize)."""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, bias)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output = self.self_attention(x, attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x
        

class DecoderBlock(nn.Module):
    """Generate output tokens once at a time while looking at: 
    encoder context representations + previous generated tokens"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, bias)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout, bias)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_output = self.self_attention(
            query=x,
            attn_mask=self_attn_mask
        )

        x = self.norm1(x + self.dropout1(self_attn_output))

        cross_attn_output = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attn_mask=cross_attn_mask
        )

        x = self.norm2(x + self.dropout2(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward():
        pass

class TransformerDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward():
        pass


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        num_layer: int,
        d_model: int,
        d_ff: int,
        d_embed: int,
        num_head: int,
        drop_out: float = 0.1,
        bias: bool = True,
    ):
        pass

    def forward(self, embed_encoder_input: torch.Tensor, embed_decoder_input: torch.Tensor, padding_mask: bool = None) -> torch.Tensor:
        """
        Args:
            embed_encoder_input:
                Shape: (batch_size, src_seq_len, d_model)
            
            embed_decoder_input:
                Shape: (batch_size, tgt_seq_len, d_model)
        """
        pass


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            src_vocab_size: Number of possible input tokens.
            tgt_vocab_size: Number of possible ouput tokens.
            num_encoder_layers: Number of encoder blocks.
            num_decoder_layers: Number of decoder blocks.
            num_heads: Number of attention heads.
            d_model: embedding/hidden size.
            d_ff: Hidden size inside feed-forward network.
            max_seq_len: Maximum sequence length for positional encoding/embedding.
            dropout: Dropout probability.


        """
        pass

    def forward():
        pass
