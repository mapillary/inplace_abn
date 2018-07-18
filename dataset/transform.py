from PIL import Image
from torchvision.transforms import functional as tfn


class SegmentationTransform:
    def __init__(self, longest_max_size, crop_size, rgb_mean, rgb_std):
        self.longest_max_size = longest_max_size
        self.crop_size = crop_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def _adjusted_scale(self, in_width, in_height):
        max_size = max(in_width, in_height)
        if max_size > self.longest_max_size:
            if max_size == in_width:
                return self.longest_max_size / in_width
            else:
                return self.longest_max_size / in_height
        else:
            return 1.

    def __call__(self, img):
        # Scaling
        scale = self._adjusted_scale(img.size[0], img.size[1])
        if scale != 1.:
            out_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(out_size, resample=Image.BILINEAR)

        valid_bbx = (0, 0, img.size[1], img.size[0])

        # Cropping / padding
        img = img.crop((0, 0, self.crop_size[1], self.crop_size[0]))

        # Convert to torch and normalize
        img = tfn.to_tensor(img)
        img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        img.div_(img.new(self.rgb_std).view(-1, 1, 1))

        return img, valid_bbx
