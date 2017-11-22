import json

DEFAULTS = {
    "network": {
        "arch": "resnet101",
        "activation": "relu",  # supported: relu, leaky_relu, elu, none
        "leaky_relu_slope": 0.01,
        "input_3x3": False,
        "bn_mode": "standard",  # supported: standard, inplace, sync
        "classes": 1000,
        "dilation": 1,
        "weight_gain_multiplier": 1,  # note: this is ignored if weight_init == kaiming_*
        "weight_init": "xavier_normal",  # supported: xavier_[normal,uniform], kaiming_[normal,uniform], orthogonal
        "devices": [0, 1, 2, 3]  # default: 4 gpus
    },
    "optimizer": {
        "batch_size": 256,
        "type": "SGD",  # supported: SGD, Adam
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "clip": 1.,
        "learning_rate": 0.1,
        "classifier_lr": -1.,  # If -1 use same learning rate as the rest of the network
        "nesterov": False,
        "schedule": {
            "type": "constant",  # supported: constant, step, multistep, exponential, linear
            "mode": "epoch",  # supported: epoch, step
            "epochs": 10,
            "params": {}
        }
    },
    "input": {
        "scale_train": -1,  # If -1 do not scale
        "crop_train": 224,
        "color_jitter_train": False,
        "lighting_train": False,
        "scale_val": 256,  # If -1 do not scale
        "crop_val": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    return config
