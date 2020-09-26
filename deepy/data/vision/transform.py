import random

import numpy as np
import cupy as cp
from cupyx.scipy.sparse import spdiags
from cupyx.scipy.sparse.linalg import lsqr

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop

from PIL import Image, ImageFilter
import kornia.filters

from deepy.data.transform import Transform, SeparatedTransform
from deepy.data.transform import PairedTransform, PairedCompose, ToPairedTransform
from ..nn import functional as myF


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


class WLSFilter(Transform):
    '''
    WLSFILTER Edge-preserving smoothing based on the weighted least squares(WLS) 
       optimization framework, as described in Farbman, Fattal, Lischinski, and
       Szeliski, "Edge-Preserving Decompositions for Multi-Scale Tone and Detail
       Manipulation", ACM Transactions on Graphics, 27(3), August 2008.
    
       Given an input image IN, we seek a new image OUT, which, on the one hand,
       is as close as possible to IN, and, at the same time, is as smooth as
       possible everywhere, except across significant gradients in L.
    
    
       Input arguments:
       ----------------
         IN              Input image (2-D, double, N-by-M matrix). 
           
         smoothness          Balances between the data term and the smoothness
                         term. Increasing smoothness will produce smoother images.
                         Default value is 1.0
           
         alpha           Gives a degree of control over the affinities by non-
                         lineary scaling the gradients. Increasing alpha will
                         result in sharper preserved edges. Default value: 1.2
           
         L               Source image for the affinity matrix. Same dimensions
                         as the input image IN. Default: log(IN)
     
    
       Example 
       -------
         RGB = imread('peppers.png'); 
         I = double(rgb2gray(RGB));
         I = I./max(I(:));
         res = wlsFilter(I, 0.5);
         figure, imshow(I), figure, imshow(res)
         res = wlsFilter(I, 2, 2);
         figure, imshow(res)
    '''
    def __init__(self, smoothness=1.0, alpha=1.2, L=None, eps=1e-8):
        super(WLSFilter, self).__init__()
        self.smoothness = smoothness
        self.alpha = alpha
        self.L = L
        self.eps = eps

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.L is None:
            L = torch.log(img + self.eps)
        else:
            L = self.L

        ch = img.size(0)
        return torch.stack([_wsl_filter(img[i], L[i], self.smoothness, self.alpha) for i in range(ch)], dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(smoothness={0}'.format(self.smoothness)
        format_string += ', alpha={0}'.format(self.alpha)
        format_string += ', L={0}'.format(self.L)
        format_string += ', eps={0}'.format(self.eps)
        format_string += ')'
        return format_string


class WLSFilterResidual(WLSFilter):
    def __init__(self, smoothness=1.0, alpha=1.2, L=None, eps=1e-8):
        super(WLSFilterResidual, self).__init__()
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        out = super(WLSFilterResidual, self).__call__(img)
        return img - out 


class GaussianBlur(torchdataset.transform.Transform):
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


class Quantize(torchdataset.transform.Transform):
    def __init__(self, n_bits=8):
        super(Quantize, self).__init__()
        self.n_bits = n_bits
        self.levels = torch.linspace(start=0., end=1., steps=(2**self.n_bits))
    
    def __call__(self, img):
        return myF.softstaircase(img, levels=self.levels, temperature=0.)
    
    def __repr__(self):
        return self.__class__.__name__ + '(n bits={0})'.format(self.n_bits)


class RandomQuantize(torchdataset.transform.Transform):
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