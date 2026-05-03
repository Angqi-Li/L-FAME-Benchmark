import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHA) mechanism with projection and dropout.
    """
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Reshape to (batch, heads, sequence, head_dim)
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)

        # Attention: (B, H, Q, K)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        # Attention output
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
    
class ResidualAdd(nn.Module):
    """
    A wrapper that applies residual connection: output = x + F(x)
    """
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    """
    Feed-forward block used after self-attention.
    Includes: Linear → GELU → Dropout → Linear.
    """
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block:
    - LayerNorm → MHA → Residual
    - LayerNorm → FFN → Residual
    """
    def __init__(self,
                 emb_size: int,
                 num_heads: int = 8,
                 drop_p: float = 0.5,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.5):
        super().__init__()
        self.attention_block = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            # DyT(emb_size),
            MultiHeadAttention(emb_size, num_heads, drop_p),
            nn.LayerNorm(emb_size),
            # DyT(emb_size),
            nn.Dropout(drop_p)
        ))
        self.feedforward_block = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            # DyT(emb_size),
            FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            # nn.LayerNorm(emb_size),
            # DyT(emb_size),
            nn.Dropout(forward_drop_p)
        ))

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention_block(x)
        x = self.feedforward_block(x)
        return x


class TransformerEncoder(nn.Sequential):
    """
    Stack of Transformer encoder blocks.
    """
    def __init__(self, num_heads: int, depth: int, emb_size: int):
        super().__init__(*[
            TransformerEncoderBlock(emb_size=emb_size, num_heads=num_heads)
            for _ in range(depth)
        ])
