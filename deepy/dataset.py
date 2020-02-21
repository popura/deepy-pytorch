import math
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import constant_
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import is_image_file, has_file_allowed_extension
from PIL import Image
import requests
import rarfile


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


def unpack_rar(source, destination=None):
    """Unpack a rar file
    """
    src_path = Path(source)
    if destination is None:
        dst_path = src_path.parent
    else:
        dst_path = Path(destination)

    rf = rarfile.RarFile(str(src_path))
    for f in rf.infolist():
        if f.isdir():
            continue

        dst_file_path = dst_path / f.filename
        print(str(dst_file_path))
        if not dst_file_path.parent.exists():
            dst_file_path.parent.mkdir(parents=True)
    
        with open(str(dst_file_path), 'wb') as unpacked_f:
            unpacked_f.write(rf.read(f))


class CaiMEImageDataset(VisionDataset):
    """Cai's multi-exposure image dataset
    """

    RESOURCES = ['1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN',
                 '16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN']

    TESTING_INDEX = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     28, 31, 33, 34, 37, 38, 39, 46, 47, 48,
                     49, 50, 51, 52, 55, 56, 57, 58, 59, 60,
                     61, 62, 63, 64, 65, 66, 67, 68, 69, 75,
                     71, 72, 73, 74, 75, 76, 77, 78, 79, 100,
                     101, 102, 103]

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
        self.samples = self._find_images(self.file_dir)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def _find_images(self, dir):
        images = []
        
        for p in sorted((self.file_dir / "target").glob("*")):
            q = self.file_dir / "multi_exposure" / p.stem
            if is_image_file(str(p)) and q.exists():
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
        root_path = Path(self.root)
        if not root_path.exists():
            root_path.mkdir(parents=True)
    
        for i in range(len(self.RESOURCES)):
            filename = root_path / ('part' + str(i) + '.rar')
            if not filename.exists():
                print('Downloading {}...'.format(str(filename)))
                download_file_from_google_drive(self.RESOURCES[i], str(filename))
            
            if not (root_path / ('Dataset_Part' + str(i+1))).exists():
                print('Unpacking {}...'.format(str(filename)))
                unpack_rar(str(filename))
        
        print('Processing downloaded dataset...')
        self.move_files()

        print('Removing downloaded raw dataset...')
        for i in range(len(self.RESOURCES)):
            (root_path / ('part' + str(i) + '.rar')).unlink()
            shutil.rmtree(root_path / ('Dataset_Part' + str(i+1)))
    
    def move_files(self):
        root_path = Path(self.root)
        modes = ['train', 'test']
        kinds = ['multi_exposure', 'target']
        dirs = ['Dataset_Part1', 'Dataset_Part2']
        
        for i in modes:
            for j in kinds:
                if (root_path / i / j).exists():
                    continue
                (root_path / i / j).mkdir(parents=True)
        
        for i, d in enumerate(dirs):
            for p in sorted((root_path / d).glob('*')):
                if not p.is_dir():
                    continue

                if str(p.name).lower() == 'label':
                    for q in sorted(p.glob('*')):
                        if i == 0 and int(q.stem) in self.TESTING_INDEX:
                            dst_path = root_path / 'test' / 'target' / ('part' + str(i+1) + '_' + q.name)
                        else:
                            dst_path = root_path / 'train' / 'target' / ('part' + str(i+1) + '_' + q.name)

                        shutil.copy2(q, dst_path)
                    continue
                
                if i == 0 and int(p.name) in self.TESTING_INDEX:
                    dst_path = root_path / 'test' / 'multi_exposure' / ('part' + str(i+1) + '_' + p.name)
                else:
                    dst_path = root_path / 'train' / 'multi_exposure' / ('part' + str(i+1) + '_' + p.name)
                
                shutil.copytree(p, dst_path)

HDR_IMG_EXTENSIONS = ('.hdr', '.exr', '.pfm')

class HDRImageFolder(VisionDataset):
    """A generic data loader where the images are arranged in this way:

    root/train/xxx.hdr
    root/train/xxy.exr
    root/train/xxz.pfm

    root/test/123.hdr
    root/test/nsdf3.exr
    root/test/asd932_.pfm

    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, transforms=None):
        super(HDRImageFolder, self).__init__(root, transforms, transform, target_transform)

        self.train = train
        if train:
            self.file_dir = Path(root) / "train"
        else:
            self.file_dir = Path(root) / "test"
        self.samples = self._find_images(self.file_dir)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def _find_images(self, dir):
        images = []
        
        for p in sorted((self.file_dir / "target").glob("*")):
            if has_file_allowed_extension(str(p), HDR_IMG_EXTENSIONS):
                images.append(str(p))
        
        return images

    def __getitem__(self, index):
        """
        TODO:
        fix it to tone map a target HDR image to an image as *img*

        """
        target_path = self.samples[index]
        target = Image.open(target_path).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.samples)
    