from dataclasses import dataclass
import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from scipy.interpolate import splprep, splev
import io
import json
import os
import copy
from flow_planner.train_utils.ddp import gather_tensor

@dataclass
class NuPlanDataSample:
    '''
    A single sample of the NuPlan dataset.
    '''
    batched: bool
    
    # input data
    ego_past: torch.Tensor = torch.tensor([])
    ego_current: torch.Tensor = torch.tensor([])
    ego_future: torch.Tensor = torch.tensor([])
    
    neighbor_past: torch.Tensor = torch.tensor([])
    neighbor_future: torch.Tensor = torch.tensor([])

    neighbor_future_observed: torch.Tensor = torch.tensor([])
    
    lanes: torch.Tensor = torch.tensor([])
    lanes_speedlimit: torch.Tensor = torch.tensor([])
    lanes_has_speedlimit: torch.Tensor = torch.tensor([])
    
    routes: torch.Tensor = torch.tensor([])
    routes_speedlimit: torch.Tensor = torch.tensor([])
    routes_has_speedlimit: torch.Tensor = torch.tensor([])
    
    map_objects: torch.Tensor = torch.tensor([])

    def gather(self, dst_rank=0):
        return NuPlanDataSample(
            batched=copy.deepcopy(self.batched),
            ego_past=torch.cat(gather_tensor(self.ego_past, dst_rank), dim=0),
            ego_current=torch.cat(gather_tensor(self.ego_current, dst_rank), dim=0),
            ego_future=torch.cat(gather_tensor(self.ego_future, dst_rank), dim=0),
            neighbor_past=torch.cat(gather_tensor(self.neighbor_past, dst_rank), dim=0),
            neighbor_future=torch.cat(gather_tensor(self.neighbor_future, dst_rank), dim=0),
            neighbor_future_observed=torch.cat(gather_tensor(self.neighbor_future_observed, dst_rank), dim=0),
            lanes=torch.cat(gather_tensor(self.lanes, dst_rank), dim=0),
            lanes_speedlimit=torch.cat(gather_tensor(self.lanes_speedlimit, dst_rank), dim=0),
            lanes_has_speedlimit=torch.cat(gather_tensor(self.lanes_has_speedlimit, dst_rank), dim=0),
            routes=torch.cat(gather_tensor(self.routes, dst_rank), dim=0),
            routes_speedlimit=torch.cat(gather_tensor(self.routes_speedlimit, dst_rank), dim=0),
            routes_has_speedlimit=torch.cat(gather_tensor(self.routes_has_speedlimit, dst_rank), dim=0),
            map_objects=torch.cat(gather_tensor(self.map_objects, dst_rank), dim=0),
        )
    
    def copy(self, device=None):
        if device is None:
            device = self.ego_past.device
        return NuPlanDataSample(
            batched=copy.deepcopy(self.batched),
            ego_past=self.ego_past.clone().to(device),
            ego_current=self.ego_current.clone().to(device),
            ego_future=self.ego_future.clone().to(device),
            neighbor_past=self.neighbor_past.clone().to(device),
            neighbor_future=self.neighbor_future.clone().to(device),
            neighbor_future_observed=self.neighbor_future_observed.clone().to(device),
            lanes=self.lanes.clone().to(device),
            lanes_speedlimit=self.lanes_speedlimit.clone().to(device),
            lanes_has_speedlimit=self.lanes_has_speedlimit.clone().to(device),
            routes=self.routes.clone().to(device),
            routes_speedlimit=self.routes_speedlimit.clone().to(device),
            routes_has_speedlimit=self.routes_has_speedlimit.clone().to(device),
            map_objects=self.map_objects.clone().to(device)
        )
        
    def to(self, target):
        """
        Moves all tensors in the data sample to the specified target.
        """
        if isinstance(target, str):
            self.ego_past = self.ego_past.to(target)
            self.ego_current = self.ego_current.to(target)
            self.ego_future = self.ego_future.to(target)
            
            self.neighbor_past = self.neighbor_past.to(target)
            self.neighbor_future = self.neighbor_future.to(target)
            self.neighbor_future_observed = self.neighbor_future_observed.to(target)

            self.lanes = self.lanes.to(target)
            self.lanes_speedlimit = self.lanes_speedlimit.to(target)
            self.lanes_has_speedlimit = self.lanes_has_speedlimit.to(target)
            
            self.routes = self.routes.to(target)
            self.routes_speedlimit = self.routes_speedlimit.to(target)
            self.routes_has_speedlimit = self.routes_has_speedlimit.to(target)
            
            self.map_objects = self.map_objects.to(target)
        else:
            self.ego_past = self.ego_past.to(target)
            self.ego_current = self.ego_current.to(target)
            self.ego_future = self.ego_future.to(target)
            
            self.neighbor_past = self.neighbor_past.to(target)
            self.neighbor_future = self.neighbor_future.to(target)
            self.neighbor_future_observed = self.neighbor_future_observed.to(target)
            
            self.lanes = self.lanes.to(target)
            self.lanes_speedlimit = self.lanes_speedlimit.to(target)
            
            self.routes = self.routes.to(target)
            self.routes_speedlimit = self.routes_speedlimit.to(target)
            
            self.map_objects = self.map_objects.to(target)

        return self
    
    def repeat(self, num_repeat):
        """
        Repeat the data sample num_repeat times.
        Returns a new instance of NuPlanDataSample with repeated data.
        """
        if self.batched:
            return NuPlanDataSample(
                batched=True,
                ego_past=self.ego_past.repeat_interleave(num_repeat, dim=0),
                ego_current=self.ego_current.repeat_interleave(num_repeat, dim=0),
                ego_future=self.ego_future.repeat_interleave(num_repeat, dim=0),
                neighbor_past=self.neighbor_past.repeat_interleave(num_repeat, dim=0),
                neighbor_future=self.neighbor_future.repeat_interleave(num_repeat, dim=0),
                neighbor_future_observed=self.neighbor_future_observed.repeat_interleave(num_repeat, dim=0),
                lanes=self.lanes.repeat_interleave(num_repeat, dim=0),
                lanes_speedlimit=self.lanes_speedlimit.repeat_interleave(num_repeat, dim=0),
                lanes_has_speedlimit=self.lanes_has_speedlimit.repeat_interleave(num_repeat, dim=0),
                routes=self.routes.repeat_interleave(num_repeat, dim=0),
                routes_speedlimit=self.routes_speedlimit.repeat_interleave(num_repeat, dim=0),
                routes_has_speedlimit=self.routes_has_speedlimit.repeat_interleave(num_repeat, dim=0),
                map_objects=self.map_objects.repeat_interleave(num_repeat, dim=0)
            )
        else:
            return NuPlanDataSample(
                batched=True,
                ego_past=self.ego_past.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                ego_current=self.ego_current.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                ego_future=self.ego_future.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                neighbor_past=self.neighbor_past.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                neighbor_future=self.neighbor_future.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                neighbor_future_observed=self.neighbor_future_observed.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                lanes=self.lanes.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                lanes_speedlimit=self.lanes_speedlimit.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                lanes_has_speedlimit=self.lanes_has_speedlimit.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                routes=self.routes.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                routes_speedlimit=self.routes_speedlimit.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                routes_has_speedlimit=self.routes_has_speedlimit.unsqueeze(0).repeat_interleave(num_repeat, dim=0),
                map_objects=self.map_objects.unsqueeze(0).repeat_interleave(num_repeat, dim=0)
            )

    def decollect(self):
        if self.batched:
            sample_list = []

            for i in range(self.ego_past.shape[0]):
                sample_list.append(NuPlanDataSample(
                    batched=False,
                    ego_past=self.ego_past[i].clone(),
                    ego_current=self.ego_current[i].clone(),
                    ego_future=self.ego_future[i].clone(),
                    
                    neighbor_past=self.neighbor_past[i].clone(),
                    neighbor_future=self.neighbor_future[i].clone(),
                    neighbor_future_observed=self.neighbor_future_observed[i].clone(),
                    
                    lanes=self.lanes[i].clone(),
                    lanes_speedlimit=self.lanes_speedlimit[i].clone(),
                    lanes_has_speedlimit=self.lanes_has_speedlimit[i].clone(),
                    
                    routes=self.routes[i].clone(),
                    routes_speedlimit=self.routes_speedlimit[i].clone(),
                    routes_has_speedlimit=self.routes_has_speedlimit[i].clone(),
                    
                    map_objects=self.map_objects[i].clone(),
                ))
            return sample_list
        else:
            return [self]
        


class NuPlanDataset(Dataset):
    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num, future_len, future_downsampling_method, max_num=None):
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len
        self._future_downsampling_method = future_downsampling_method

        self.fail_token = []

        self.data_list = self.data_list if max_num is None else self.data_list[:max_num]

    def __len__(self):
        return len(self.data_list)

    def downsample_future_data(self, origin_future_data):

        if self._future_downsampling_method == "uniform":
            sample_rate = origin_future_data.shape[0] // self._future_len
            start_index = sample_rate - 1
            sub_data = origin_future_data[start_index:]

            sampled_data = sub_data[::sample_rate]

            return sampled_data
        elif self._future_downsampling_method == "log":
            # sample future state from time [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 8.0]
            assert self._future_len == 8, f"log sampling only suitable for future len = 8, but got {self._future_len}"
            sampled_data = origin_future_data[[0, 1, 3, 7, 15, 31, 63, 79]]

            return sampled_data
        else:
            raise ValueError(f"future downsampling method only supports: [uniform, log], but got {self._future_downsampling_method}")


    def generate_new_index(self):
        return random.randint(0, len(self) - 1)


    def __getitem__(self, idx) -> NuPlanDataSample:
        # while True:
        #     try:
        #         data_path = os.path.join(self.data_dir, self.data_list[idx])
        #         data = opendata(data_path)
        #         break
        #         # data = np.load(self.data_list[idx])
        #     except:
        #         self.fail_token.append(self.data_list[idx])
        #         dump({'fail_file': self.fail_token}, './fail.json', file_format='json', indent=4)
        #         idx = self.generate_new_index()
        data = np.load(os.path.join(self.data_dir, self.data_list[idx]))
        ego_agent_past = torch.from_numpy(data['ego_agent_past'])
        ego_current_state = torch.from_numpy(data['ego_current_state'])
        ego_agent_future = torch.from_numpy(data['ego_agent_future']).to(torch.float32)

        neighbor_agents_past = torch.from_numpy(data['neighbor_agents_past'][:self._past_neighbor_num])
        neighbor_agents_future = torch.from_numpy(data['neighbor_agents_future'][:self._predicted_neighbor_num])
        neighbor_future_observed = torch.from_numpy(data['neighbor_agents_future'])

        lanes = torch.from_numpy(data['lanes'])
        lanes_speed_limit = torch.from_numpy(data['lanes_speed_limit'])
        lanes_has_speed_limit = torch.from_numpy(data['lanes_has_speed_limit'])

        route_lanes = torch.from_numpy(data['route_lanes'])
        route_lanes_speed_limit = torch.from_numpy(data['route_lanes_speed_limit'])
        route_lanes_has_speed_limit = torch.from_numpy(data['route_lanes_has_speed_limit'])

        static_objects = torch.from_numpy(data['static_objects'])

        data = NuPlanDataSample(
            batched=False,
            ego_past=ego_agent_past,
            ego_current=ego_current_state,
            ego_future=ego_agent_future,
            neighbor_past=neighbor_agents_past,
            neighbor_future=neighbor_agents_future,
            neighbor_future_observed=neighbor_future_observed,
            lanes=lanes,
            lanes_speedlimit=lanes_speed_limit,
            lanes_has_speedlimit=lanes_has_speed_limit,
            routes=route_lanes,
            routes_speedlimit=route_lanes_speed_limit,
            routes_has_speedlimit=route_lanes_has_speed_limit,
            map_objects=static_objects
        )

        return data

def openjson(path):
    with open(path, "r") as f:
        dict = json.loads(f.read())
    return dict

def opendata(path):
    npz_data = np.load(path)

    return npz_data
