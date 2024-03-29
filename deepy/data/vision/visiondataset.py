import os
import os.path

from PIL import Image

from .. import dataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
HDR_IMG_EXTENSIONS = ('.hdr', '.exr', '.pfm')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def hdr_loader(path):
    raise NotImplementedError


class ImageFolder(dataset.DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None, transforms=None,
                 pre_load=False, pre_transform=None, pre_target_transform=None, pre_transforms=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          transforms=transforms,
                                          pre_load=pre_load,
                                          pre_transform=pre_transform,
                                          pre_target_transform=pre_target_transform,
                                          pre_transforms=pre_transforms,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


class UnorganizedImageFolder(dataset.UnorganizedDatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, transforms=None,
                 pre_load=False, pre_transform=None, pre_target_transform=None, pre_transforms=None,
                 loader=default_loader, is_valid_file=None):

        super(UnorganizedImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                     transform=transform,
                                                     target_transform=target_transform,
                                                     transforms=transforms,
                                                     pre_load=pre_load,
                                                     pre_transform=pre_transform,
                                                     pre_target_transform=pre_target_transform,
                                                     pre_transforms=pre_transforms,
                                                     is_valid_file=is_valid_file)
        self.imgs = self.samples