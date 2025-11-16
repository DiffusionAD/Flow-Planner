import torch
from flow_planner.core.common.json_utils import openjson

class StateNormalizer:
    
    def __init__(self, config, future_downsampling_method, predicted_neighbor_num):
        
        self.config = config
        self.future_downsampling_method = future_downsampling_method
        self.predicted_neighbor_num = predicted_neighbor_num
        
        mean = [[config['ego'][future_downsampling_method]['mean']]] + [[config['neighbor'][future_downsampling_method]['mean']]] * self.predicted_neighbor_num
        self.mean = torch.as_tensor(mean)
        
        std = [[config['ego'][future_downsampling_method]['std']]] + [[config['neighbor'][future_downsampling_method]['std']]] * self.predicted_neighbor_num
        self.std = torch.as_tensor(std)
        
    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist()
        }