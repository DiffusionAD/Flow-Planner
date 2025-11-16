from dataclasses import dataclass
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

        
class MaskedTensor(torch.Tensor):
    data: torch.Tensor
    mask: torch.Tensor
    
    def __new__(cls, data, mask=None):
        """
        创建一个新的 MaskedTensor 实例。
        Args:
            data (torch.Tensor): 张量数据。
            mask (torch.Tensor): 掩码，与 data 的形状相同。
        """
        # 创建基础 Tensor
        instance = super().__new__(cls, data.shape, dtype=data.dtype, device=data.device)
        instance.data = data
        instance.mask = mask if mask is not None else torch.ones_like(data, dtype=torch.bool)
        return instance

    def __repr__(self):
        return f"MaskedTensor(data={self.data}, mask={self.mask})"

    def apply_mask(self):
        """将掩码应用于数据，返回被掩码覆盖的数据（无效部分为 0）。"""
        return self.data * self.mask

    def __torch_function__(self, func, types, args=(), kwargs=None):
        logger.warning(f"Please implement any tensor operator manually for MaskedTensor.")
        return NotImplemented
        # """
        # 重载所有 PyTorch 张量操作，确保 mask 和 data 的一致性。
        # """
        # if kwargs is None:
        #     kwargs = {}

        # # 检查是否处理 MaskedTensor
        # if not all(issubclass(t, torch.Tensor) for t in types):
        #     return NotImplemented

        # # 执行原函数
        # result = func(*args, **kwargs)

        # # 如果返回的是张量，保持掩码一致
        # if isinstance(result, torch.Tensor):
        #     if func in {torch.add, torch.mul, torch.sub, torch.div}:
        #         # 对于数学操作，掩码取交集
        #         new_mask = torch.logical_and(*(arg.mask if isinstance(arg, MaskedTensor) else torch.ones_like(result, dtype=torch.bool) for arg in args))
        #         return MaskedTensor(result, mask=new_mask)
        #     elif func in {torch.stack, torch.cat}:
        #         tensors = args[0]
        #         if all(isinstance(t, MaskedTensor) for t in tensors):
        #             data = func([t.data for t in tensors], *args[1:], **kwargs)
        #             mask = func([t.mask for t in tensors], *args[1:], **kwargs)
        #             return MaskedTensor(data, mask)
        #     else:
        #         # 默认情况下触发未定义
        #         logger.warning(f"Operation {func} is not supported for MaskedTensor, returning NotImplemented.")
        #         return NotImplemented
            
        # return result

    def __getitem__(self, idx):
        """
        支持切片操作，同时更新掩码。
        """
        data = self.data[idx]
        mask = self.mask[idx]
        return MaskedTensor(data, mask)

    def clone(self):
        """
        支持克隆 MaskedTensor，包括其数据和掩码。
        """
        return MaskedTensor(self.data.clone(), self.mask.clone())