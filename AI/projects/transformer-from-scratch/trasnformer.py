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
            bias: bool = True,
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
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
            value: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """
        Args:
            query: (batch, query_len, d_model)
            key: (batch, key_len, d_model). Defaults to query for self-attention.
            value: (batch, key_len, d_model). Defaults to key.
            attention_mask: mask broadcastable to (batch, num_heads, query_len, key_len).
        """
        key = query if key is None else key
        value = key if value is None else value

        # TODO: Project query, key, and value with q_proj, k_proj, and v_proj.
        # TODO: Split dimensions into (batch, num_heads, seq_len, head_dim).
        # TODO: Compute QK^T / sqrt(head_dim).
        # TODO: Apply attention_mask before softmax.
        # TODO: Apply softmax and attention_dropout, then multiply by V.
        # TODO: Merge heads back to (batch, query_len, d_model).
        # TODO: Return the result of out_proj.
        raise NotImplementedError("Complete the multi-head attention forward pass")


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward():
        pass


class EncoderBlock(nn.Module):
    """Convert source tokens to context-aware representations (Contextualize)."""
    def __init__():
        pass

    def forward():
        pass


class DecoderBlock(nn.Module):
    """Generate output tokens once at a time while looking at: 
    encoder context representations + previous generated tokens"""
    def __init__():
        pass

    def forward():
        pass


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
