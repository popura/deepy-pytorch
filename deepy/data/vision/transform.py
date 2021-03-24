import random

import numpy as np

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop

from PIL import Image, ImageFilter
import kornia.filters

from deepy.data.transform import Transform, SeparatedTransform
from deepy.data.transform import PairedTransform, PairedCompose, ToPairedTransform
from deepy.nn import functional as myF


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class PowerTwoResize(Transform):
    def __init__(self, interpolation=Image.BILINEAR):
        super(PowerTwoResize, self).__init__()
        self.interpolation = interpolation

    def __call__(self, img):
        height = img.height
        width = img.width
        new_size = (int(2**np.floor(np.log2(height))), int(2**np.floor(np.log2(width))))
        return F.resize(img, new_size, self.interpolation)

    def __repr__(self, ):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(interpolation={0})'.format(interpolate_str)


class ResizeToMultiple(Transform):
    def __init__(self, divisor, interpolation=Image.BILINEAR):
        super().__init__()
        self.divisor = int(divisor)
        self.interpolation = interpolation

    def __call__(self, img):
        new_size = ((img.height // self.divisor) * self.divisor,
                    (img.width // self.divisor) * self.divisor)
        return F.resize(img, new_size, self.interpolation)

    def __repr__(self, ):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(interpolation={0})'.format(interpolate_str)


class PairedRandomHorizontalFlip(PairedTransform):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, target):
        if random.random() > self.p:
            return F.hflip(img), F.hflip(target)
        return img, target
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class PairedPowerTwoResize(PairedTransform):
    def __init__(self, interpolation=Image.BILINEAR):
        super(PairedPowerTwoResize, self).__init__()
        self.interpolation = interpolation

    def __call__(self, img, target):
        if not img.size == target.size:
            raise ValueError('Size of img and target must be the same')

        height = img.height
        width = img.width
        new_size = (int(2**np.floor(np.log2(height))), int(2**np.floor(np.log2(width))))
        return F.resize(img, new_size, self.interpolation), F.resize(target, new_size, self.interpolation)

    def __repr__(self, ):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(interpolation={0})'.format(interpolate_str)


class PairedRandomResizedCrop(PairedTransform):
    def __init__(self, size, scale=(0.08, 1.0),
                 ratio=(3./4., 4./3.), interpolation=Image.BILINEAR):
        self.transform = RandomResizedCrop(size, scale,
                                           ratio, interpolation)
    
    def __call__(self, img, target):
        i, j, h, w = self.transform.get_params(img, self.transform.scale,
                                               self.transform.ratio)
        transformed_img = F.resized_crop(img, i, j, h, w, self.transform.size,
                                         self.transform.interpolation)
        transformed_target = F.resized_crop(target, i, j, h, w, self.transform.size,
                                            self.transform.interpolation)
        return transformed_img, transformed_target
    
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class GaussianBlur(Transform):
    """Filter image with a Gaussian kernel

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        Image tensor: Blurred version of the input.

    """

    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.filter = kornia.filters.GaussianBlur2d(kernel_size, sigma)

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): Image to be blurred with a size of C * H * W

        Returns:
            torch.Tensor: blurred image.
        """
        return self.filter(img.unsqueeze(0)).squeeze(0)

    def __repr__(self):
        return self.__class__.__name__ + '(kernel size={0})'.format(self.kernel_size)


class GaussianBlurResidual(GaussianBlur):
    def __init__(self, kernel_size, sigma):
        super(GaussianBlurResidual, self).__init__(kernel_size, sigma)
    
    def __call__(self, img):
        out = super(GaussianBlurResidual, self).__call__(img)
        return img - out 


class Quantize(Transform):
    def __init__(self, n_bits=8):
        super(Quantize, self).__init__()
        self.n_bits = n_bits
        self.levels = torch.linspace(start=0., end=1., steps=(2**self.n_bits))
    
    def __call__(self, img):
        return myF.softstaircase(img, levels=self.levels, temperature=0.)
    
    def __repr__(self):
        return self.__class__.__name__ + '(n bits={0})'.format(self.n_bits)


class RandomQuantize(Transform):
    def __init__(self, min_val=0.25, max_val=4, n_bits=8):
        super(RandomQuantize, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.n_bits = n_bits
        self.quantizer = Quantize(n_bits)
    
    def __call__(self, img):
        ratio = random.uniform(self.min_val, self.max_val)
        img *= ratio
        img = self.quantizer(img)
        img /= ratio
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '(min={}, max={}, n bits={})'.format(self.min_val, self.max_val, self.n_bits)
