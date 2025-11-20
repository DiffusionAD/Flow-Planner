from __future__ import annotations

import torch
from torch import Tensor
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import pack, unpack
from einops.layers.torch import Rearrange


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

class BiasedAttention(Module):
    def __init__(
        self,
        dropout = 0.,
    ):
        super().__init__()
        # dropouts
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        q, k, v,
        mask = None,
        taus = None,
        attn_bias = None,
    ):
        D = q.shape[-1]

        sim = einsum('bhid, bhjd -> bhij', q, k) / (D**0.5)

        if mask is not None:
            mask = mask[:, None, None, :]
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if attn_bias is not None:
            attn_bias = taus * attn_bias.unsqueeze(-1) # (B, token_num, token_num, heads)
            attn_bias = attn_bias.permute(0, 3, 1, 2) 
            sim = sim - attn_bias # (B, heads, token_num, token_num)

        attn = F.softmax(sim, dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)
        attn = self.attn_dropout(attn)
        out = einsum('bhij, bhjd -> bhid', attn, v)

        return out

class JointAttention(Module):
    def __init__(
        self,
        dim_inputs: tuple[int, ...],
        dim_head = 64,
        heads = 8,
        enable_attn_dist = False,
        token_num = 118,
        attend_kwargs: dict = dict(),
    ):
        super().__init__()
        """
        Global attention of inputs features.
        Reference implementation: https://github.com/lucidrains/mmdit
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.in_proj = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = BiasedAttention(
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.out_proj = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.token_num = token_num

        if enable_attn_dist:
            self.gen_taus = nn.Linear(dim_head * heads, self.token_num * heads)
        else:
            self.gen_taus = None

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: tuple[Tensor],
        masks: tuple[Tensor | None] | None = None,
        attn_dist = None,
    ):

        device = self.dummy.device

        B = inputs[0].shape[0]

        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv in zip(inputs, masks, self.in_proj):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)
            all_qkvs.append(qkv)

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)
            all_masks.append(mask)

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        q, k, v = all_qkvs
        
        taus = None if self.gen_taus is None else self.gen_taus(q.reshape(B, self.token_num, -1)).reshape(B, self.token_num, self.token_num, -1)

        outs = self.attend(q=q, k=k, v=v, mask=all_masks, taus=taus, attn_bias=attn_dist)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.out_proj):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)