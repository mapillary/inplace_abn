from random import sample

import torch

# Default augmentation values compatible with ImageNet data augmentation pipeline
_DEFAULT_ALPHASTD = 0.1
_DEFAULT_EIGVAL = [0.2175, 0.0188, 0.0045]
_DEFAULT_EIGVEC = [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
_DEFAULT_BCS = [0.4, 0.4, 0.4]


def _grayscale(img):
    alpha = img.new([0.299, 0.587, 0.114])
    return (alpha.view(3, 1, 1) * img).sum(0, keepdim=True)


def _blend(img1, img2, alpha):
    return img1 * alpha + (1 - alpha) * img2


class Lighting:
    def __init__(self, alphastd=_DEFAULT_ALPHASTD, eigval=_DEFAULT_EIGVAL, eigvec=_DEFAULT_EIGVEC):
        self._alphastd = alphastd
        self._eigval = eigval
        self._eigvec = eigvec

    def __call__(self, img):
        if self._alphastd == 0.:
            return img

        alpha = torch.normal(img.new_zeros(3), self._alphastd)
        eigval = img.new(self._eigval)
        eigvec = img.new(self._eigvec)

        rgb = (eigvec * alpha * eigval).sum(dim=1)
        return img + rgb.view(3, 1, 1)


class Saturation(object):
    def __init__(self, var):
        self._var = var

    def __call__(self, img):
        gs = _grayscale(img)
        alpha = img.new(1).uniform_(-self._var, self._var) + 1.0
        return _blend(img, gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self._var = var

    def __call__(self, img):
        gs = torch.zeros_like(img)
        alpha = img.new(1).uniform_(-self._var, self._var) + 1.0
        return _blend(img, gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self._var = var

    def __call__(self, img):
        gs = _grayscale(img)
        gs = img.new_full((1, 1, 1), gs.mean())
        alpha = img.new(1).uniform_(-self._var, self._var) + 1.0
        return _blend(img, gs, alpha)


class ColorJitter(object):
    def __init__(self, saturation=_DEFAULT_BCS[0], brightness=_DEFAULT_BCS[1], contrast=_DEFAULT_BCS[2]):
        self._transforms = []
        if saturation is not None:
            self._transforms.append(Saturation(saturation))
        if brightness is not None:
            self._transforms.append(Brightness(brightness))
        if contrast is not None:
            self._transforms.append(Contrast(contrast))

    def __call__(self, img):
        if len(self._transforms) == 0:
            return img

        for t in sample(self._transforms, len(self._transforms)):
            img = t(img)
        return img
