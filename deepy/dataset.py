import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import constant_
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import is_image_file
from PIL import Image

class CaiMEImageDataset(VisionDataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, transforms=None):
        super(CaiMEImageDataset, self).__init__(root, transforms, transform, target_transform)
        self.train = train
        if train:
            self.file_dir = Path(root) / "train"
        else:
            self.file_dir = Path(root) / "test"
        self.samples = self._find_images(self.file_dir)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def _find_images(self, dir):
        images = []
        
        for p in sorted((self.file_dir / "target").glob("*")):
            q = self.file_dir / "multi_exposure" / p.stem
            if is_image_file(str(p)) and q.exists:
                for r in q.glob("*"):
                    if is_image_file(str(r)):
                        item = (str(r), str(p))
                        images.append(item)
        
        return images

    def __getitem__(self, index):
        exposures_path, target_path = self.samples[index]
        img = Image.open(exposures_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.samples)