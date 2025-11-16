import torch
import wandb
import os
import copy
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import sys
from flow_planner.recorder import RecorderBase

class TensorboardRecorder(RecorderBase):
    
    def __init__(self, save_path, rank=0):
        super().__init__()
        self.writer = None
        self.rank = int(rank)
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=save_path)


    def record_loss(self, loss: Dict, step: int):
        if self.writer is not None:
            for key, value in loss.items():
                self.writer.add_scalar(key, value, step)
        
    def record_metric(self, metrics: Dict, step: int):
        """
        metrics (dict):
        step (int, optional): epoch or step
        """
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)