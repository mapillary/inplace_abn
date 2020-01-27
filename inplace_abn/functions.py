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
    def _gather_values(*tensors, group, world_size):
        # Start gather operations asynchronously
        gathered, gather_ops = [], []
        for t in tensors:
            t_all = t.new_empty(world_size, *t.shape)
            t_op = distributed.all_gather(list(t_all.unbind(0)), t, group=group, async_op=True)

            gathered.append(t_all)
            gather_ops.append(t_op)

        # Wait
        for op in gather_ops:
            op.wait()

        # Return results
        return tuple(gathered)

    @staticmethod
    def _reduce_forward(mean, var, count, group, world_size):
        all_mean, all_var, all_count = InPlaceABN._gather_values(
            mean, var, count, group=group, world_size=world_size)
        return _backend.reduce_statistics(all_mean, all_var, all_count)

    @staticmethod
    def _reduce_backward(sum_dy, sum_xhat_dy, group, world_size):
        all_sum_dy, all_sum_xhat_dy = InPlaceABN._gather_values(
            sum_dy, sum_xhat_dy, group=group, world_size=world_size)
        return all_sum_dy.sum(dim=0), all_sum_xhat_dy.sum(dim=0)

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation="leaky_relu", activation_param=0.01,
                group=None):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = _activation_from_name(activation)
        ctx.activation_param = activation_param
        ctx.group = group

        # Check if we really need to perform distributed operations
        if ctx.group is not None:
            ctx.distributed = True
            ctx.world_size = distributed.get_world_size(group=group)
        else:
            ctx.distributed = False
            ctx.world_size = 1

        if ctx.training:
            mean, var, count = _backend.statistics(x)

            # Gather stats from all workers if needed
            if ctx.distributed:
                mean, var, count = InPlaceABN._reduce_forward(mean, var, count, ctx.group, ctx.world_size)

            # Update running stats
            count_ = count.to(dtype=var.dtype)
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * count_ / (count_ - 1))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var, count = running_mean, running_var, None

            # Mark in-place modified tensors
            ctx.mark_dirty(x)

        # Transform x
        _backend.forward(x, mean, var, weight, bias, ctx.eps, ctx.activation, ctx.activation_param)

        # Save for backward
        ctx.save_for_backward(x, var, count, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dy_act):
        y_act, var, count, weight, bias = ctx.saved_tensors

        # Call backward_reduce if we need to compute at least one of the gradients
        if any(ctx.needs_input_grad):
            xhat, dy, sum_dy_local, sum_xhat_dy_local = _backend.backward_reduce(
                y_act, dy_act, weight, bias, ctx.eps, ctx.activation, ctx.activation_param)

            if ctx.distributed:
                sum_dy, sum_xhat_dy = InPlaceABN._reduce_backward(
                    sum_dy_local, sum_xhat_dy_local, ctx.group, ctx.world_size)
            else:
                sum_dy, sum_xhat_dy = sum_dy_local, sum_xhat_dy_local
        else:
            return None, None, None, None, None, None, None, None, None, None

        # Gradient w.r.t. x
        if ctx.needs_input_grad[0]:
            if ctx.training:
                # This overwrites dy with dx
                _backend.backward_train(xhat, dy, var, count, sum_dy, sum_xhat_dy, weight, ctx.eps)
                dx = dy
            else:
                dx = _backend.backward_test(dy_act, var, weight, ctx.eps)
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

        return dx, dweight, dbias, None, None, None, None, None, None, None, None


def inplace_abn(x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation="leaky_relu", activation_param=0.01):
    return InPlaceABN.apply(x, weight, bias, running_mean, running_var,
                            training, momentum, eps, activation, activation_param, None)


def inplace_abn_sync(x, weight, bias, running_mean, running_var,
                     training=True, momentum=0.1, eps=1e-05, activation="leaky_relu", activation_param=0.01,
                     group=distributed.group.WORLD):
    return InPlaceABN.apply(x, weight, bias, running_mean, running_var,
                            training, momentum, eps, activation, activation_param, group)


__all__ = ["inplace_abn", "inplace_abn_sync"]
