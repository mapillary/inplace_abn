# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional, Any
from warnings import warn

import torch
import torch.autograd as autograd
import torch.distributed as distributed
from torch.autograd.function import once_differentiable

from . import _backend


def _activation_from_name(activation):
    if activation == "leaky_relu":
        return _backend.Activation.LeakyReLU
    elif activation == "elu":
        return _backend.Activation.ELU
    elif activation == "identity":
        return _backend.Activation.Identity
    else:
        raise ValueError("Unknown activation function {}".format(activation))


def _count_samples(x):
    count = x.size(0)
    for i in range(2, x.ndimension()):
        count *= x.size(i)
    return count


class InPlaceABN(autograd.Function):
    @staticmethod
    def _reduce_forward(mean, var, count, group, world_size):
        # Mean and variance
        mean_var = torch.cat([mean, var], dim=0)
        all_mean_var = mean_var.new_empty(world_size, mean_var.numel())
        distributed.all_gather(
            list(all_mean_var.unbind(0)), mean_var, group=group, async_op=False
        )
        all_mean, all_var = all_mean_var.split(mean.numel(), dim=1)

        # Count
        all_count = count.new_empty(world_size, 1)
        distributed.all_gather(
            list(all_count.unbind(0)), count, group=group, async_op=False
        )

        return _backend.reduce_statistics(all_mean, all_var, all_count)

    @staticmethod
    def _reduce_backward(sum_dy, sum_xhat_dy, group):
        stacked = torch.cat([sum_dy, sum_xhat_dy], dim=0)
        distributed.all_reduce(
            stacked, distributed.ReduceOp.SUM, group=group, async_op=False
        )
        return torch.split(stacked, sum_dy.numel(), dim=0)

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        running_mean,
        running_var,
        training=True,
        momentum=0.1,
        eps=1e-05,
        activation="leaky_relu",
        activation_param=0.01,
        group=None,
        world_size=1,
    ):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = _activation_from_name(activation)
        ctx.activation_param = activation_param
        ctx.group = group
        ctx.world_size = world_size
        ctx.has_running_stats = running_mean is not None and running_mean is not None

        if ctx.training:
            mean, var, count = _backend.statistics(x)

            # Gather stats from all workers if needed
            if ctx.world_size > 1:
                mean, var, count = InPlaceABN._reduce_forward(
                    mean, var, count, ctx.group, ctx.world_size
                )

            # Update running stats if needed
            if ctx.has_running_stats:
                count_ = count.to(dtype=var.dtype)
                running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
                running_var.mul_((1 - ctx.momentum)).add_(
                    ctx.momentum * var * count_ / (count_ - 1)
                )
        else:
            mean, var, count = running_mean, running_var, None

        # Transform x
        _backend.forward(
            x, mean, var, weight, bias, ctx.eps, ctx.activation, ctx.activation_param
        )

        # Save for backward and mark dirty tensors
        ctx.save_for_backward(x, var, count, weight, bias)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dy_act):
        y_act, var, count, weight, bias = ctx.saved_tensors
        # Call backward_reduce if we need to compute at least one of the gradients
        if any(ctx.needs_input_grad):
            # remove memory overlaping to allow for in-place operation
            dy_act = dy_act.contiguous()
            # This overwrites y_act with xhat and dy_act with dy
            xhat, dy, sum_dy_local, sum_xhat_dy_local = _backend.backward_reduce(
                y_act,
                dy_act,
                weight,
                bias,
                ctx.eps,
                ctx.activation,
                ctx.activation_param,
            )

            if ctx.world_size > 1:
                sum_dy, sum_xhat_dy = InPlaceABN._reduce_backward(
                    sum_dy_local, sum_xhat_dy_local, ctx.group
                )
            else:
                sum_dy, sum_xhat_dy = sum_dy_local, sum_xhat_dy_local
        else:
            return (None,) * 12

        # Gradient w.r.t. x
        if ctx.needs_input_grad[0]:
            if ctx.training:
                # This overwrites dy with dx
                _backend.backward_train(
                    xhat, dy, var, count, sum_dy, sum_xhat_dy, weight, ctx.eps
                )
                dx = dy
            else:
                dx = _backend.backward_test(dy, var, weight, ctx.eps)
        else:
            dx = None

        # Gradient w.r.t. weight
        if weight is not None and ctx.needs_input_grad[1]:
            dweight = sum_xhat_dy_local
            dweight[weight < 0] *= -1
        else:
            dweight = None

        # Gradient w.r.t. bias
        if bias is not None and ctx.needs_input_grad[2]:
            dbias = sum_dy_local
        else:
            dbias = None

        return (dx, dweight, dbias) + (None,) * 9


def inplace_abn(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
    activation: str = "leaky_relu",
    activation_param: float = 0.01,
):
    """InPlace Activated Batch Normalization

    This applies the following per-channel combined BatchNorm + activation operation:

        x_hat = (x - mu) / sqrt(sigma^2 + eps)
        x <- act(x_hat, p) * (|weight| + eps) + bias

    where:
        - mu is the per-channel batch mean, or `running_mean` if `training` is `False`
        - sigma^2 is the per-channel batch variance, or `running_var` if `training` is `False`
        - act(., p) is the activation function specified by `activation`
        - p is `activation_param`, i.e. the negative slope of Leaky ReLU or alpha
          parameter of ELU
        - `weight` and `bias` are the optional affine parameters
        - `eps` is a small positive number

    The running statistics, if given and if `training` is `True` are updated as follows:

        running_mean <- running_mean * momentum + (1 - momentum) * mu
        running_var <- running_var * momentum + (1 - momentum) * unbiased_sigma^2

    where unbiased_sigma^2 is the unbiased batch variance

    Args:
        x: Input tensor with shape N x C or N x C x S_1 x ... x S_n, which will be
            overwritten with the result
        weight: Tensor of affine scale parameters with shape C, or `None`
        bias: Tensor of affine bias parameters with shape C, or `None`
        running_mean: Running mean tensor with shape C, or `None`
        running_var: Running variance tensor with shape C, or `None`
        training: If `True` compute, use and update batch statistics, otherwise use
            running statistics
        momentum: Momentum factor applied to compute running statistics
        eps: Small constant to prevent numerical issues
        activation: Name of the activation function, one of: `leaky_relu`, `elu` or `identity`
        activation_param: Negative slope for the `leaky_relu` activation or `alpha`
            parameter for the `elu` activation
    """
    if training:
        samples = _count_samples(x)
        if samples <= 1:
            raise ValueError(
                "inplace_abn is trying to compute batch statistics, but the input "
                "tensor only contains a single sample per channel"
            )

    return InPlaceABN.apply(
        x,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        activation,
        activation_param,
        None,
        1,
    )


def inplace_abn_sync(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
    activation: str = "leaky_relu",
    activation_param: float = 0.01,
    group: Optional[Any] = None,
):
    """InPlace Activated Batch Normalization with distributed synchronization

    This operates like `inplace_abn`, but assumes to be called by all replicas
    in the given distributed group, and computes batch statistics across all of them.
    Note that the input tensors can have different dimensions in each replica.

    Args:
        x: Input tensor with shape N x C or N x C x S_1 x ... x S_n, which will be
            overwritten with the result
        weight: Tensor of affine scale parameters with shape C, or `None`
        bias: Tensor of affine bias parameters with shape C, or `None`
        running_mean: Running mean tensor with shape C, or `None`
        running_var: Running variance tensor with shape C, or `None`
        training: If `True` compute, use and update batch statistics, otherwise use
            running statistics
        momentum: Momentum factor applied to compute running statistics
        eps: Small constant to prevent numerical issues
        activation: Name of the activation function, one of: `leaky_relu`, `elu` or `identity`
        activation_param: Negative slope for the `leaky_relu` activation or `alpha`
            parameter for the `elu` activation
        group: Distributed group to synchronize with, or `None` to use the default group
    """
    if training:
        samples = _count_samples(x)
        if samples <= 1:
            raise ValueError(
                "inplace_abn_sync is trying to compute batch statistics, but the input "
                "tensor only contains a single sample per channel"
            )

    if distributed.is_initialized():
        if group is None:
            group = distributed.group.WORLD
        world_size = distributed.get_world_size(group)

        return InPlaceABN.apply(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            activation,
            activation_param,
            group,
            world_size,
        )
    else:
        warn(
            "inplace_abn_sync is being called, but torch.distributed is not initialized. "
            "Reverting to non-synchronized inplace_abn.",
            category=RuntimeWarning,
        )

        return InPlaceABN.apply(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            activation,
            activation_param,
            None,
            1,
        )


__all__ = ["inplace_abn", "inplace_abn_sync"]
