"""Minimal pre-norm transformer encoder used as the optional temporal trunk."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    """Scaled dot-product attention.

    Args:
        q, k, v : (N, H, T, d_head)
        mask    : optional (N, T) bool key-padding mask, True = ignore that key.
    """
    d_k = q.size(-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        attn_logits = attn_logits.masked_fill(
            mask.unsqueeze(1).unsqueeze(2), torch.finfo(attn_logits.dtype).min
        )

    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (N, H, T, 3*d_head)
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o


class EncoderBlock(nn.Module):
    """Pre-norm encoder block: x + Attn(LN(x)), then x + FFN(LN(x)).

    Pre-norm is more stable to train than post-norm, especially without LR
    warm-up. Because the residual stream is never normalised, the stack needs a
    final LayerNorm; ``model.TransformerTrunk`` applies it.
    """

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.linear_net(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        """Per-layer attention maps, stacked as (num_layers, N, H, T, T)."""
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(layer.norm1(x), mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x, mask=mask)
        return torch.stack(attention_maps)
