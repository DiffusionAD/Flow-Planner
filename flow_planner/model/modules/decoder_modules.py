import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath
from flow_planner.model.model_utils.tool_func import modulate

# RMSNorm from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

# FeedForward from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mlp_ratio = 4.0,
        dropout = 0.,
        activation = None,
        bias = True,
    ):
        super().__init__()
        inner_dim = int(dim * mlp_ratio)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU() if activation is None else activation

        self.ffn = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = bias),
            activation,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = bias),
        )

    def muon_parameters(self):
        weights = []

        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue

            weights.append(m.weight)

        return weights

    def forward(self, x):
        return self.ffn(x)

# Adaptive LayerNorm module from https://github.com/lucidrains/mmdit/blob/main/mmdit/mmdit_generalized_pytorch.py
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, dim_cond = None
    ):
        super().__init__()
        has_cond = dim_cond is not None
        self.has_cond = has_cond

        self.ln = nn.LayerNorm(dim, elementwise_affine = not has_cond)
        
        if has_cond:
            cond_linear = nn.Linear(dim_cond, dim * 2)

            self.to_cond = nn.Sequential(
                # Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)
            nn.init.constant_(cond_linear.bias[:dim], 1.)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(
        self,
        x,
        cond = None
    ):
        assert not ((cond is not None) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        x = self.ln(x)

        if self.has_cond:
            gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
            x = x * gamma + beta

        return x

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    def forward(self, x):
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class PostFusion(nn.Module):
    def __init__(self, hidden_dim, heads, action_num, mlp_hidden_dim=256, dropout=0.1):
        '''
        Further fusing the tokens after MMDiT.
        '''
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.action_num = action_num
        self.attend = SelfAttentionBlock(dim=hidden_dim, heads=heads)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_token, kv_token, kv_mask):
        total_token = torch.cat([kv_token, x_token], dim=1)

        total_mask = torch.cat([kv_mask, torch.ones_like(kv_mask[:, :self.action_num], dtype=bool, device=kv_mask.device)], dim=-1)
        fused_token = self.attend(total_token, ~total_mask)
        pooled_token = fused_token.mean(dim=1)[:, None, :]
        x_token = x_token + self.mlp(self.norm(pooled_token))
        return x_token
    

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_dim, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )


    def forward(self, x, y):
        B, P, _ = x.shape
        
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)
        
        x = modulate(self.norm_final(x), shift, scale)
        return self.proj(x)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        t_emb = self.mlp(t_freq)
        return t_emb