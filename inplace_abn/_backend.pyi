"""Stubs for the native methods"""
from typing import Tuple, Optional

import torch


def statistics(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def reduce_statistics(all_mean: torch.Tensor, all_var: torch.Tensor, all_count: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def forward(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor,
            weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
            eps: float, activation, activation_param: float) -> None: ...


def backward_reduce(y_act: torch.Tensor, dy_act: torch.Tensor, weight: Optional[torch.Tensor],
                    bias: Optional[torch.Tensor], eps: float, activation, activation_param: float) \
        -> Tuple[torch.Tensor, torch.Tensor]: ...


def backward_train(xhat: torch.Tensor, dy: torch.Tensor, var: torch.Tensor, count: torch.Tensor, sum_dy: torch.Tensor,
                   sum_xhat_dy: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor: ...


def backward_test(dy: torch.Tensor, var: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor: ...


class Activation:
    LeakyReLU = ...
    ELU = ...
    Identity = ...
