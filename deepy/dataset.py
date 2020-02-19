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
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


class CaiMEImageDataset(VisionDataset):
    """Cai's multi-exposure image dataset
    """

    RESOURCES = ['1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN',
                 '16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN']

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, transforms=None, download=False):
        super(CaiMEImageDataset, self).__init__(root, transforms, transform, target_transform)

        if download:
            self.download()
        
        self.train = train
        if train:
            self.file_dir = Path(root) / "train"
        else:
            self.file_dir = Path(root) / "test"
        self.samples = self._find_images(self, self.file_dir)
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
    
    def download(self):
        p = Path(self.root)
        if not p.exests:
            p.mkdir(parents=True)
    
        for i in range(len(self.RESOURCES)):
            download_file_from_google_drive(self.RESOURCES[i], self.root)