import torch
import numpy as np

class NuPlanAugment():

    def __init__(self, state_aug, map_aug, device):
        self.state_aug = state_aug
        self.map_aug = map_aug
        self.device = device

    def __call__(self, data):
        # modify for mutable object
        if self.state_aug is not None:
            data = self.state_aug(data)
        return data