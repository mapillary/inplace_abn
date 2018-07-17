from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from modules import ABN, InPlaceABN, InPlaceABNSync
from .transforms import ColorJitter, Lighting


def _get_norm_act(network_config):
    if network_config["bn_mode"] == "standard":
        assert network_config["activation"] in ("relu", "leaky_relu", "elu", "none"), \
            "Standard batch normalization is only compatible with relu, leaky_relu, elu and none"
        activation_fn = partial(ABN,
                                activation=network_config["activation"],
                                slope=network_config["leaky_relu_slope"])
    elif network_config["bn_mode"] == "inplace":
        assert network_config["activation"] in ("leaky_relu", "elu", "none"), \
            "Inplace batch normalization is only compatible with leaky_relu, elu and none"
        activation_fn = partial(InPlaceABN,
                                activation=network_config["activation"],
                                slope=network_config["leaky_relu_slope"])
    elif network_config["bn_mode"] == "sync":
        assert network_config["activation"] in ("leaky_relu", "elu", "none"), \
            "Sync batch normalization is only compatible with leaky_relu, elu and none"
        activation_fn = partial(InPlaceABNSync,
                                activation=network_config["activation"],
                                slope=network_config["leaky_relu_slope"],
                                devices=network_config["devices"])
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
        transforms.RandomSizedCrop(input_config["crop_train"]),
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
        val_transforms.append(transforms.Scale(input_config["scale_val"]))
    val_transforms += [
        transforms.CenterCrop(input_config["crop_val"]),
        transforms.ToTensor(),
        normalize,
    ]

    return train_transforms, val_transforms
