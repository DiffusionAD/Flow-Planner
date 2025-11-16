import torch
import torch.nn as nn
from torch import Tensor
from timm.layers import Mlp
from flow_planner.model.modules.decoder_modules import FinalLayer, PostFusion
from flow_planner.model.model_utils.tool_func import sinusoidal_positional_encoding
from flow_planner.model.modules.decoder_modules import RMSNorm, FeedForward, AdaptiveLayerNorm
from flow_planner.model.flow_planner_model.global_attention import JointAttention


class FlowPlannerDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim,
            depth,
            t_embedder,
            agents_hidden_dim=192,
            lane_hidden_dim=192,
            heads=6,
            preproj_hidden=256,
            enable_attn_dist=False,
            act_pe_type: str = 'learnable',
            device: str = 'cuda',
            **planner_params
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = planner_params['action_len'] * planner_params['state_dim']
        action_num = (planner_params['future_len'] - planner_params['action_overlap']) // (planner_params['action_len'] - planner_params['action_overlap'])
        self.action_num = int(action_num)
        self.token_num = planner_params['neighbor_num'] + planner_params['static_num'] + planner_params['lane_num'] + self.action_num
        self.action_len = planner_params['action_len']
        self.action_overlap = planner_params['action_overlap']
        self.state_dim = planner_params['state_dim']
        
        self.dit = FlowPlannerDiT(
            depth=depth,
            dim_modalities=(
                agents_hidden_dim,
                lane_hidden_dim,
                hidden_dim
            ),
            dim_cond=hidden_dim,
            heads=heads,
            dim_head=int(hidden_dim/heads),
            enable_attn_dist=enable_attn_dist,
            token_num=self.token_num
        )
        
        self.post_fusion = PostFusion(hidden_dim=hidden_dim, heads=heads, action_num=self.action_num)
        self.t_embedder = t_embedder        
        self.preproj = Mlp(in_features=self.output_dim, hidden_features=preproj_hidden, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.final_layer = FinalLayer(hidden_dim, self.output_dim)

        self.cfg_embedding = nn.Embedding(2, hidden_dim) # an embedding that indicates if the neighbor vehicles are dropped
        self.act_pe_type = act_pe_type
        self.load_action_pe(act_pe_type) # load action positional embedding into the model
        
        self.agents_in_proj = nn.Linear(agents_hidden_dim, hidden_dim) if agents_hidden_dim != hidden_dim \
            else nn.Identity()
        self.lane_in_proj = nn.Linear(lane_hidden_dim, hidden_dim) if lane_hidden_dim != hidden_dim \
            else nn.Identity()
        
        self.planner_params = planner_params
        
        self.device = device
        
        self.initialize_weights()

    def load_action_pe(self, act_pe_type: str):
        if act_pe_type == 'learnable':
            self.action_pe = nn.Parameter(torch.Tensor(self.action_num, self.hidden_dim))
            nn.init.normal_(self.action_pe, mean=0.0, std=1.0)
        elif act_pe_type == 'fixed_sin':
            action_t = (torch.arange(0, self.action_num) * (self.action_len - self.action_overlap) + self.action_len / 2)
            action_pe = sinusoidal_positional_encoding(action_t, self.hidden_dim)
            self.register_buffer('action_pe', action_pe)
        elif act_pe_type == 'none':
            action_pe = torch.zeros((self.action_num, self.hidden_dim))
            self.register_buffer('action_pe', action_pe)
        else:
            raise ValueError(f'Unexpected action embedding type {act_pe_type}')
    
    def initialize_weights(self):
        # TODO: complete initialization
        def basic_init(module):
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
        self.apply(basic_init)
    
    def forward(self, x, t, **model_extra):
        '''
        Forward pass of the FlowPlannerDecoder.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)                  -> Time step
        model_extra: dict containing additional information for the model, such as:
            - 'encodings': list of encodings for different modalities, each with shape (B
            , neighbor, lane, static, hidden_dim)
            - 'token_dist': attention distribution for the tokens, shape (B, neighbor, lane, static, token_num)
            - 'masks': list of masks for different modalities, each with shape (B, neighbor, lane, static, token_num)
            - 'cfg_flags': tensor of shape (B,) indicating whether to use configuration flags
            - 'routes_cond': tensor of shape (B, hidden_dim) for route conditions
            - 'obs_cond': tensor of shape       (B, neighbor, lane, static, hidden_dim) for observation conditions  
        '''
        B, P, _, _ = x.shape
        x = x.to(torch.float32)
        x = x.reshape(B, P, -1)
        encodings = list(model_extra['encodings']) # (neighbor, lane, static)

        attn_dist = model_extra['token_dist']

        masks = list(model_extra['masks'])

        cfg_flags = model_extra['cfg_flags'].reshape(B) # (B,)
        cfg_embedding = self.cfg_embedding(cfg_flags).unsqueeze(-2) # (B, 1, hidden_dim)

        routes_cond = model_extra['routes_cond']

        x = self.preproj(x)
        encodings.append(x)
        masks.append(None)
        
        time_cond = self.t_embedder(t).unsqueeze(1)
        routes_cond = routes_cond.unsqueeze(1)        
        action_pe = self.action_pe.unsqueeze(0).repeat(B, 1, 1) # concate an extra dimension for path token

        y = time_cond + routes_cond + action_pe + cfg_embedding

        token_tuple = self.dit(
            modality_tokens=encodings, 
            modality_masks=masks, 
            time_cond=time_cond, 
            routes_cond=routes_cond, 
            action_encoding=action_pe, 
            cfg_embedding=cfg_embedding, 
            attn_dist=attn_dist
        )
        
        agents_token, lanes_token, x_token = token_tuple

        agents_token = self.agents_in_proj(agents_token)
        lanes_token = self.lane_in_proj(lanes_token)

        kv_token = torch.cat([agents_token, lanes_token], dim=1)
        key_mapping_mask = torch.cat([masks[i] for i in range(len(masks)-1)], dim=1) # here the valid mask is represented as True (1)

        x_token = self.post_fusion(x_token, kv_token, ~key_mapping_mask)

        prediction = self.final_layer(x_token, y)

        prediction = prediction.reshape(B, P, -1, self.planner_params['state_dim'])

        return prediction

class FlowPlannerDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim_modalities: tuple[int, ...],
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        enable_attn_dist = False,
        ff_kwargs: dict = dict(),
        token_num: int = 118,
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        self.modalities_gate_proj = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_cond, dim * 2)
            )
        for dim in dim_modalities])

        for layer in self.modalities_gate_proj:
            nn.init.zeros_(layer[-1].weight)
            nn.init.constant_(layer[-1].bias, 1.)

        self.attn_layernorms = nn.ModuleList([AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities])

        self.joint_attn = JointAttention(
            dim_inputs = dim_modalities,
            dim_head = dim_head,
            heads = heads,
            enable_attn_dist = enable_attn_dist,
            token_num=token_num
        )

        self.ff_layernorms = nn.ModuleList([AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities])
        self.feedforwards = nn.ModuleList([FeedForward(dim, **ff_kwargs) for dim in dim_modalities])

    def forward(
        self,
        *,
        modality_tokens,
        modality_masks = None,
        modality_conds = None,
        attn_dist = None
    ):
        assert len(modality_tokens) == self.num_modalities
 
        attn_gammas = []
        ff_gammas = []
        for proj, cond in zip(self.modalities_gate_proj, modality_conds):
            gamma = proj(cond)
            attn_g, ff_g = gamma.chunk(2, dim=-1)
            attn_gammas.append(attn_g)
            ff_gammas.append(ff_g)
        
        modality_tokens_attn_res = [token.clone() for token in modality_tokens]
        modality_tokens = [ln(tokens, cond=ln_cond) for ln, tokens, ln_cond in zip(self.attn_layernorms, modality_tokens, modality_conds)]
        modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks, attn_dist = attn_dist)
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, attn_gammas)]
        modality_tokens = [token + res for token, res in zip(modality_tokens, modality_tokens_attn_res)]

        modality_tokens_ffn_res = [token.clone() for token in modality_tokens]
        modality_tokens = [ln(tokens, cond=ln_cond) for ln, tokens, ln_cond in zip(self.ff_layernorms, modality_tokens, modality_conds)]
        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, ff_gammas)]
        modality_tokens = [token + res for token, res in zip(modality_tokens, modality_tokens_ffn_res)]

        return modality_tokens

class FlowPlannerDiT(nn.Module):
    def __init__(
        self,
        *,
        depth, # TODO
        dim_modalities, # TODO
        enable_attn_dist = False,
        **block_kwargs
    ):
        super().__init__()

        blocks = [FlowPlannerDiTBlock(dim_modalities = dim_modalities, enable_attn_dist = enable_attn_dist, **block_kwargs) for _ in range(depth)]
        self.blocks = nn.ModuleList(blocks)

        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = nn.ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens,
        modality_masks = None,
        time_cond = None,
        routes_cond = None,
        action_encoding = None,
        cfg_embedding = None,
        attn_dist = None,
    ):
        other_modality_conds = [time_cond + routes_cond] * (len(modality_tokens) - 1)
        ego_traj_conds = [time_cond + routes_cond + action_encoding + cfg_embedding]
        modality_conds = other_modality_conds + ego_traj_conds

        for block in self.blocks:
            modality_tokens = block(
                modality_tokens = modality_tokens,
                modality_masks = modality_masks,
                modality_conds = modality_conds,
                attn_dist = attn_dist
            )

        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]
        
        return tuple(modality_tokens)