from typing import Optional

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as functional

from .functions import inplace_abn, inplace_abn_sync


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Args:
        num_features: Number of feature channels in the input and output
        eps: Small constant to prevent numerical issues
        momentum: Momentum factor applied to compute running statistics with
            exponential moving average, or `None` to compute running statistics
            with cumulative moving average
        affine: If `True` apply learned scale and shift transformation after normalization
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and uses batch statistics instead
            in both training and eval modes if the running mean and variance are `None`
        activation: Name of the activation functions, one of: `relu`, `leaky_relu`,
            `elu` or `identity`
        activation_param: Negative slope for the `leaky_relu` activation or `alpha`
            parameter for the `elu` activation
    """

    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "num_features",
        "affine",
        "activation",
        "activation_param",
    ]
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    activation: str
    activation_param: float

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
    ):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.activation = activation
        self.activation_param = activation_param
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _get_momentum_and_training(self):
        if self.momentum is None:
            momentum = 0.0
        else:
            momentum = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    momentum = 1.0 / float(self.num_batches_tracked)
                else:
                    momentum = self.momentum

        if self.training:
            training = True
        else:
            training = (self.running_mean is None) and (self.running_var is None)

        return momentum, training

    def _get_running_stats(self):
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )
        return running_mean, running_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        momentum, training = self._get_momentum_and_training()
        running_mean, running_var = self._get_running_stats()

        x = functional.batch_norm(
            x,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            training,
            momentum,
            self.eps,
        )

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(
                x, negative_slope=self.activation_param, inplace=True
            )
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError(f"Unknown activation function {self.activation}")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(ABN, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        rep = "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}"
        if self.activation in ["leaky_relu", "elu"]:
            rep += "[{activation_param}]"
        return rep.format(**self.__dict__)


class InPlaceABN(ABN):
    """InPlace Activated Batch Normalization

    Args:
        num_features: Number of feature channels in the input and output
        eps: Small constant to prevent numerical issues
        momentum: Momentum factor applied to compute running statistics with
            exponential moving average, or `None` to compute running statistics
            with cumulative moving average
        affine: If `True` apply learned scale and shift transformation after normalization
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and uses batch statistics instead
            in both training and eval modes if the running mean and variance are `None`
        activation: Name of the activation functions, one of: `relu`, `leaky_relu`,
            `elu` or `identity`
        activation_param: Negative slope for the `leaky_relu` activation or `alpha`
            parameter for the `elu` activation
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
    ):
        super(InPlaceABN, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            activation,
            activation_param,
        )

    def forward(self, x):
        momentum, training = self._get_momentum_and_training()
        running_mean, running_var = self._get_running_stats()

        return inplace_abn(
            x,
            self.weight,
            self.bias,
            running_mean,
            running_var,
            training,
            momentum,
            self.eps,
            self.activation,
            self.activation_param,
        )


class InPlaceABNSync(ABN):
    """InPlace Activated Batch Normalization with distributed synchronization

    This operates like `inplace_abn`, but assumes to be called by all replicas
    in a given distributed group, and computes batch statistics across all of them.
    Note that the input tensors can have different dimensions in each replica.

    Args:
        num_features: Number of feature channels in the input and output
        eps: Small constant to prevent numerical issues
        momentum: Momentum factor applied to compute running statistics with
            exponential moving average, or `None` to compute running statistics
            with cumulative moving average
        affine: If `True` apply learned scale and shift transformation after normalization
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and uses batch statistics instead
            in both training and eval modes if the running mean and variance are `None`
        activation: Name of the activation functions, one of: `relu`, `leaky_relu`,
            `elu` or `identity`
        activation_param: Negative slope for the `leaky_relu` activation or `alpha`
            parameter for the `elu` activation
        group: Distributed group to synchronize with, default is WORLD
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
        group=distributed.group.WORLD,
    ):
        super(InPlaceABNSync, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            activation,
            activation_param,
        )
        self.group = group

    def set_group(self, group):
        """Set distributed group to synchronize with

        This function should never be called between forward and backward

        Args:
            group: The new distributed group to synchronize with
        """
        self.group = group

    def forward(self, x):
        momentum, training = self._get_momentum_and_training()
        running_mean, running_var = self._get_running_stats()

        return inplace_abn_sync(
            x,
            self.weight,
            self.bias,
            running_mean,
            running_var,
            training,
            momentum,
            self.eps,
            self.activation,
            self.activation_param,
            self.group,
        )
