from typing import Union, List, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

def collect_batch(data: Union[List[Dict[str, Any]], List[object]]) -> Dict[str, Any]:
    """
    Collect a batch of data samples into a single dictionary.
    Args:
        data (List[Dict[str, Any]]): A list of data samples.
    Returns:
        Dict[str, Any]: A dictionary containing the batched data.
    """
    if isinstance(data[0], dict):
        batch = {}
        for key in data[0].keys():
            if isinstance(data[0][key], torch.Tensor):
                batch[key] = torch.stack([sample[key] for sample in data], dim=0)
            else:
                raise ValueError(f"Unsupported data type: {type(data[0][key])}")
    elif isinstance(data[0], object):
        batch = {}
        for key in data[0].__dict__.keys():
            if isinstance(data[0].__dict__[key], torch.Tensor):
                batch[key] = torch.stack([getattr(sample, key) for sample in data], dim=0)
            elif key == "batched":
                batch[key] = True
            else:
                raise ValueError(f"Unsupported data type: {type(data[0].__dict__[key])}")
        batch = data[0].__class__(**batch)
    return batch