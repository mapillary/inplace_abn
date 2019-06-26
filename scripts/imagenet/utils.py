import time
from functools import partial

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from inplace_abn import ABN, InPlaceABN, InPlaceABNSync
from .transforms import ColorJitter, Lighting


def _get_norm_act(network_config):
    if network_config["bn_mode"] == "standard":
        assert network_config["activation"] in ("relu", "leaky_relu", "elu", "identity"), \
            "Standard batch normalization is only compatible with relu, leaky_relu, elu and identity"
        activation_fn = partial(ABN,
                                activation=network_config["activation"],
                                activation_param=network_config["activation_param"])
    elif network_config["bn_mode"] == "inplace":
        assert network_config["activation"] in ("leaky_relu", "elu", "identity"), \
            "Inplace batch normalization is only compatible with leaky_relu, elu and identity"
        activation_fn = partial(InPlaceABN,
                                activation=network_config["activation"],
                                activation_param=network_config["activation_param"])
    elif network_config["bn_mode"] == "sync":
        assert network_config["activation"] in ("leaky_relu", "elu", "identity"), \
            "Sync batch normalization is only compatible with leaky_relu, elu and identity"
        activation_fn = partial(InPlaceABNSync,
                                activation=network_config["activation"],
                                activation_param=network_config["activation_param"])
    else:
        print("Unrecognized batch normalization mode", network_config["bn_mode"])
        exit(1)

    return activation_fn


def get_model_params(network_config):
    """Convert a configuration to actual model parameters

    Parameters
    ----------
    network_config : dict
        Dictionary containing the configuration options for the network.

    Returns
    -------
    model_params : dict
        Dictionary containing the actual parameters to be passed to the `net_*` functions in `models`.
    """
    model_params = {}
    if network_config["input_3x3"] and not network_config["arch"].startswith("wider"):
        model_params["input_3x3"] = True
    model_params["norm_act"] = _get_norm_act(network_config)
    model_params["classes"] = network_config["classes"]
    if not network_config["arch"].startswith("wider"):
        model_params["dilation"] = network_config["dilation"]
    return model_params


def create_optimizer(optimizer_config, model):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.

    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    if optimizer_config["classifier_lr"] != -1:
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if k.find("fc") != -1:
                classifier_params.append(v)
            else:
                net_params.append(v)
        params = [
            {"params": net_params},
            {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
        ]
    else:
        params = model.parameters()

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              weight_decay=optimizer_config["weight_decay"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    if optimizer_config["schedule"]["type"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":
        def linear_lr(it):
            return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler


def create_transforms(input_config):
    """Create transforms from configuration

    Parameters
    ----------
    input_config : dict
        Dictionary containing the configuration options for input pre-processing.

    Returns
    -------
    train_transforms : list
        List of transforms to be applied to the input during training.
    val_transforms : list
        List of transforms to be applied to the input during validation.
    """
    normalize = transforms.Normalize(mean=input_config["mean"], std=input_config["std"])

    train_transforms = []
    if input_config["scale_train"] != -1:
        train_transforms.append(transforms.Scale(input_config["scale_train"]))
    train_transforms += [
        transforms.RandomResizedCrop(input_config["crop_train"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if input_config["color_jitter_train"]:
        train_transforms.append(ColorJitter())
    if input_config["lighting_train"]:
        train_transforms.append(Lighting())
    train_transforms.append(normalize)

    val_transforms = []
    if input_config["scale_val"] != -1:
        val_transforms.append(transforms.Resize(input_config["scale_val"]))
    val_transforms += [
        transforms.CenterCrop(input_config["crop_val"]),
        transforms.ToTensor(),
        normalize,
    ]

    return train_transforms, val_transforms


def create_test_transforms(config, crop, scale, ten_crops):
    normalize = transforms.Normalize(mean=config["mean"], std=config["std"])

    val_transforms = []
    if scale != -1:
        val_transforms.append(transforms.Resize(scale))
    if ten_crops:
        val_transforms += [
            transforms.TenCrop(crop),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack(crops))
        ]
    else:
        val_transforms += [
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize
        ]

    return val_transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_sum(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0))
    return res


def validate(val_loader, model, criterion, ten_crops=False, print_freq=1, it=None, tb=None, logger=print):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    do_print = rank == 0

    def process(input, target, all_reduce=None):
        with torch.no_grad():
            if ten_crops:
                bs, ncrops, c, h, w = input.size()
                input = input.view(-1, c, h, w)

            target = target.cuda(non_blocking=True)

            # compute output
            if ten_crops:
                output = model(input).view(bs, ncrops, -1).mean(1)
            else:
                output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy_sum(output.data, target, topk=(1, 5))

            loss *= target.shape[0]
            count = target.new_tensor([target.shape[0]], dtype=torch.long)
            if all_reduce:
                all_reduce(count)
            for meter, val in (losses, loss), (top1, prec1), (top5, prec5):
                if all_reduce:
                    all_reduce(val)
                val /= count.item()
                meter.update(val.item(), count.item())

    # deal with remainder
    all_reduce = partial(dist.all_reduce, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
    last_group_size = len(val_loader.dataset) % world_size
    for i, (input, target) in enumerate(val_loader):
        if input.shape[0] > 1 or last_group_size == 0:
            process(input, target, all_reduce)
        else:
            process(input, target,
                    partial(dist.all_reduce, op=dist.ReduceOp.SUM, group=dist.new_group(range(last_group_size))))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if do_print and i % print_freq == 0:
            logger('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f}) \t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    if input.shape[0] == 1 and rank > last_group_size > 0:
        dist.new_group(range(last_group_size))

    if do_print:
        logger(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
               .format(top1=top1, top5=top5))

    if it is not None and (not dist.is_initialized() or dist.get_rank() == 0):
        tb.add_scalar("val/loss", losses.avg, it)
        tb.add_scalar("val/top1", top1.avg, it)
        tb.add_scalar("val/top5", top5.avg, it)

    return top1.avg
