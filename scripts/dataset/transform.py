from PIL import Image
from torchvision.transforms import functional as tfn


class SegmentationTransform:
    def __init__(self, longest_max_size, rgb_mean, rgb_std):
        self.longest_max_size = longest_max_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def __call__(self, img):
        # Scaling
        scale = self.longest_max_size/float(max(img.size[0],img.size[1]))
        if scale != 1.:
            out_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(out_size, resample=Image.BILINEAR)

        # Convert to torch and normalize
        img = tfn.to_tensor(img)
        img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        img.div_(img.new(self.rgb_std).view(-1, 1, 1))

        return img
