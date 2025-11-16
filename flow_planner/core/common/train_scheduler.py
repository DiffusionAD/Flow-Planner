import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, MultiplicativeLR


def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1):
    assert epoch >= warm_up_epoch
    T_warmup = warm_up_epoch
    
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    scheduler = SequentialLR(optimizer, 
                             schedulers=[warmup_scheduler, fixed_scheduler], 
                             milestones=[T_warmup])
    
    return scheduler