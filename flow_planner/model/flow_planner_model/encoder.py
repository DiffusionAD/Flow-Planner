import torch
from torch import nn
from flow_planner.model.modules.encoder_modules import *

class FlowPlannerEncoder(nn.Module):
    def __init__(self, 
                 encoder_hidden_dim,
                 with_ego_history,

                 neighbor_encoder: AgentFusionEncoder,
                 static_encoder: StaticFusionEncoder,
                 lane_encoder: LaneFusionEncoder,
                 route_encoder: RouteEncoder,

                 action_length: int,
                 action_overlap: int,

                 static_objects_num=5,
                 future_len: int=80,

                 lane_num=70, # 70
                 lane_dim=12, # 12
                 neighbor_agent_num=32,
                 neighbor_pred_num=10,
                 ):
        super().__init__()

        self.with_ego_history = with_ego_history
        self.lane_dim = lane_dim

        self.neighbor_encoder = neighbor_encoder

        self.static_encoder = static_encoder
        
        self.lane_encoder = lane_encoder

        self.route_encoder = route_encoder

        self.future_len = future_len
        self.neighbor_agent_num = neighbor_agent_num
        self.static_num = static_objects_num
        self.lane_num = lane_num
        self.neighbor_pred_num = neighbor_pred_num

        self.token_num = self.neighbor_agent_num + self.static_num + self.lane_num

        self.pos_emb = nn.Linear(7, encoder_hidden_dim)
        self.hidden_dim = encoder_hidden_dim

        action_num = (self.future_len - action_overlap) // (action_length - action_overlap)
        self.action_num = int(action_num)

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize embedding MLP:
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, neighbors, static, lanes, lanes_speed_limit, lanes_has_speed_limit, routes):
        
        B = neighbors.shape[0]
        # neighbor_embedding = self.type_embedding.weight[0][None, None, :].expand(B, self.neighbor_num, -1)
        # static_embedding = self.type_embedding.weight[1][None, None, :].expand(B, self.static_num, -1)
        # lane_embedding = self.type_embedding.weight[2][None, None, :].expand(B, lanes.shape[1], -1)

        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

        lanes_loc = lanes[:, :, int(self.lane_encoder._lane_points_num / 2), :2].clone()
        static_loc = static[:, :, :2].clone()
        neighbors_loc = neighbors[:, :, -1, :2].clone()
        ego_loc = torch.tensor([-0.5, 0], device=neighbors.device)[None, None, :].repeat(B, self.action_num, 1)
        pred_neighbor_loc = neighbors[:, :self.neighbor_pred_num, -1, :2].clone()
        all_loc = torch.cat([neighbors_loc, static_loc, lanes_loc, ego_loc, pred_neighbor_loc], dim=-2)
        token_dist = torch.norm(all_loc[:, None, :, :] - all_loc[:, :, None, :], dim=-1)

        def encoding_process(encoding, mask, pos):
            token_num = encoding.shape[1]
            pos = pos.view(B * token_num, -1).type(torch.float32)
            mask = mask.view(-1)
            encoding_pos = self.pos_emb(pos[~mask])
            encoding_pos_result = torch.zeros((B * token_num, self.hidden_dim), device=encoding_pos.device)
            encoding_pos_result[~mask] = encoding_pos
            encoding = encoding + encoding_pos_result.view(B, token_num, -1)
            return encoding
        
        neighbors_encoding = encoding_process(encoding_neighbors, neighbors_mask, neighbor_pos)
        static_encoding = encoding_process(encoding_static, static_mask, static_pos)
        lanes_encoding = encoding_process(encoding_lanes, lanes_mask, lane_pos)

        routes_cond = self.route_encoder(routes)
        
        encoder_outputs=dict(encodings=(torch.cat([neighbors_encoding, static_encoding], dim=1),
                                        lanes_encoding),
                              masks=(torch.cat([~neighbors_mask, ~static_mask], dim=1), ~lanes_mask),
                              routes_cond=routes_cond,
                              token_dist=token_dist)

        return encoder_outputs