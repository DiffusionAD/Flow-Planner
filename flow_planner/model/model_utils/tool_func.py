import torch
import math
import numpy as np
from flow_planner.data.dataset.nuplan import NuPlanDataSample

def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x

def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale) + shift, x_rest], dim=1)
    else:
        x = x * (1 + scale) + shift

    return x

def sinusoidal_positional_encoding(pos_tensor: torch.Tensor, d_model: int) -> torch.Tensor:

    if d_model % 2 != 0:
        raise ValueError("The feature dimension must be even for sinusoidal positional encoding")

    pos = pos_tensor.float()

    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=pos.device).float() *
        (-math.log(100.0) / d_model)
    )

    pos_encodings = pos.unsqueeze(-1) * div_term

    pe = torch.zeros(*pos.shape, d_model, device=pos.device)
    pe[..., 0::2] = torch.sin(pos_encodings)
    pe[..., 1::2] = torch.cos(pos_encodings)

    return pe

def lanes_to_route_mask(lanes: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    """
    lanes:  (B, N, L, D)
    routes: (B, M, L, D)

    Returns:
      mask (B, N) of dtype torch.uint8 (or bool) where mask[b,i] == 1
      iff lane i in batch b exactly matches one of the M routes in that batch,
      and both the lane and the matched route are non-zero (i.e. not all padding).
    """
    # sums over the (L,D) dims to detect all-zero (invalid) entries
    lane_sum  = lanes.abs().sum(dim=(2,3))   # (B, N)
    route_sum = routes.abs().sum(dim=(2,3))  # (B, M)

    lane_valid  = lane_sum  > 0              # (B, N)
    route_valid = route_sum > 0              # (B, M)

    # compare every lane to every route in batch:
    #  lanes.unsqueeze(2): (B, N, 1, L, D)
    #  routes.unsqueeze(1): (B, 1, M, L, D)
    # => broadcasted to (B, N, M, L, D)
    eq = torch.isclose(
        lanes.unsqueeze(2),
        routes.unsqueeze(1)
    ).all(dim=(3,4))                         # (B, N, M)  True where lane==route

    # ignore comparisons to “invalid” routes
    eq = eq & route_valid.unsqueeze(1)       # (B, N, M)

    # if any route matches a lane, we mark that lane as selected
    selected = eq.any(dim=2)                 # (B, N)

    # but any all-zero (invalid) lane must remain unselected
    selected = selected & lane_valid         # (B, N)

    return selected.to(torch.int64)          # or .to(torch.bool)