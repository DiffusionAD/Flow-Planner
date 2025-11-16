import torch
import math
from flow_planner.data.dataset.nuplan import NuPlanDataSample

class ModelInputProcessor:
    def __init__(
        self,
        future_len,
        obs_normalizer,
        state_normalizer,
        neighbor_pred_num
    ):
        self.future_len = future_len
        self.obs_normalizer = obs_normalizer
        self.state_normalizer = state_normalizer
        self.neighbor_pred_num = neighbor_pred_num

    def state_preprocess(self, x):
        return self.state_normalizer(x) if self.state_normalizer is not None else x
    
    def state_postprocess(self, x):
        return self.state_normalizer.inverse(x) if self.state_normalizer is not None else x

    def x_differentiate(self, x_future, x_current):
        x_all = torch.cat([x_current, x_future], dim=-2)
        return x_all[..., 1:, :] - x_all[..., :-1, :]

    def x_integral(self, dx_future, x_current):
        v_all = torch.cat([x_current, dx_future], dim=-2)
        return torch.cumsum(v_all, dim=-2)[..., 1:, :]

    def sample_to_model_input(
        self,
        data: NuPlanDataSample,
        device,
        kinematic,
        is_training: bool=False
    ):
                
        if self.obs_normalizer is not None:
            model_inputs = self.obs_normalizer(data)

        ego_future = data.ego_future
        if ego_future.numel() != 0:
            ego_future = ego_future[..., 1:1+self.future_len, :3] # (x, y, heading)

        # in the original input, the neighbor_future only include 10 neighbors, while the neighbor_agent_current include 32 neighbors
        
        neighbor_future = data.neighbor_future
        if neighbor_future.numel() != 0:
            neighbor_future_mask = torch.sum(torch.ne(neighbor_future[..., :self.future_len, :3],0), dim=-1) == 0
            neighbor_future = neighbor_future[..., :self.future_len, :3] # (x, y, heading)

            neighbor_future[neighbor_future_mask] = 0.
            neighbor_future_valid = ~neighbor_future_mask

        model_inputs = {}
        model_inputs['ego_past'] = data.ego_past.to(device)
        model_inputs['neighbor_past'] = data.neighbor_past.to(device)
        model_inputs['lanes'] = data.lanes.to(device)
        model_inputs['lanes_speedlimit'] = data.lanes_speedlimit.to(device)
        model_inputs['lanes_has_speedlimit'] = data.lanes_has_speedlimit.to(device)
        model_inputs['routes'] = data.routes.to(device)
        model_inputs['map_objects'] = data.map_objects.to(device)
        
        neighbor_current = data.neighbor_past[..., :self.neighbor_pred_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbor_current[..., :4], 0), dim=-1) == 0
        model_inputs['neighbor_current_mask'] = neighbor_current_mask.to(device)
        

        ego_current_state = data.ego_current
        model_inputs['ego_current'] = ego_current_state
        ego_current_xy_cos_sin = ego_current_state[..., :4]
        ego_current = torch.cat([
            ego_current_xy_cos_sin[..., :2],
            torch.atan2(ego_current_xy_cos_sin[..., 3:4], ego_current_xy_cos_sin[..., 2:3])
        ], dim=-1)
        neighbor_current = torch.cat([
            neighbor_current[..., :2],
            torch.atan2(neighbor_current[..., 3:4], neighbor_current[..., 2:3])
        ], dim=-1)
        current_states = torch.cat([
            ego_current[:, None],
            neighbor_current
        ], dim=1)
        
        if is_training:
            gt_future = torch.cat([
                ego_future[:, None, :, :],
                neighbor_future,
            ], dim=1)

            gt_with_current = torch.cat([
                    current_states[:, :, None, :],
                    gt_future
                ], dim=2)
            
            neighbor_mask = torch.cat([
                    neighbor_current_mask.unsqueeze(-1),
                    neighbor_future_mask
            ], dim=-1)
            gt_with_current[:, 1:][neighbor_mask] = 0.

            gt_with_current.to(device)
            neighbor_future_valid.to(device)

        else:
            gt_with_current = current_states[:, :, None, :].repeat(1, 1, self.future_len + 1, 1)
            neighbor_future_valid = None

        if kinematic == 'waypoints':
            gt_with_current = torch.cat([
                gt_with_current[..., :2],
                torch.cat([
                    gt_with_current[..., 2:3].cos(),
                    gt_with_current[..., 2:3].sin()
                ], dim=-1)
            ], dim=-1)
        elif kinematic == 'velocity':
            future_velocity = self.x_differentiate(gt_with_current[..., 1:, :], gt_with_current[..., :1, :])
            gt_with_current = torch.cat([gt_with_current[..., :1, :], future_velocity], dim=-2)
        elif kinematic == 'acceleration':
            future_velocity = self.x_differentiate(gt_with_current[..., 1:, :], gt_with_current[..., :1, :])
            current_velocity = torch.cat([ego_current_state[..., 4:6], ego_current_state[..., 9:10]], dim=-1)[:, None, None, :]
            future_acc = self.x_differentiate(future_velocity, current_velocity)
            gt_with_current = torch.cat([current_velocity, future_acc], dim=-2)
        
        gt_with_current = self.state_preprocess(gt_with_current)
            
        model_inputs.update({'neighbor_future_valid': neighbor_future_valid})

        return model_inputs, gt_with_current