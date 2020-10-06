import unittest

import torch

import deepy.data
import deepy.data.vision
from deepy.data import ToyClassDataset, ToyRegDataset


class TestClassDataset(unittest.TestCase):

    def test_inheritance(self):
        ds = ToyClassDataset(num_classes=7)
        self.assertIsInstance(ds, torch.utils.data.Dataset)


class TestRegDataset(unittest.TestCase):

    def test_inheritance(self):
        ds = ToyRegDataset()
        self.assertIsInstance(ds, torch.utils.data.Dataset)


class TestPureDatasetFolder(unittest.TestCase):
    pass


class TestUnorganizedDatasetFolder(unittest.TestCase):
    pass


class TestDatasetFolder(unittest.TestCase):
    pass


class TestPickleFolder(unittest.TestCase):
    pass


class TestSelfSupervisedDataset(unittest.TestCase):
    
    def test_init(self):
        ds = ToyRegDataset()
        ss = deepy.data.SelfSupervisedDataset(ds)
        self.assertIsInstance(ss, deepy.data.SelfSupervisedDataset)
        self.assertIsInstance(ss, torch.utils.data.Dataset)

    def test_len(self):
        ds = ToyRegDataset()
        ss = deepy.data.SelfSupervisedDataset(ds)
        self.assertEqual(len(ss), len(ds))
    
    def test_getitem(self):
        ds = ToyRegDataset()
        ss = deepy.data.SelfSupervisedDataset(ds)
        sample1, *_ = ds[0]
        sample2, target2 = ss[0]
        self.assertEqual(sample2, target2)

        sample2, target2 = ss[10]
        self.assertEqual(sample2, 10)


class TestInverseDataset(unittest.TestCase):

    def test_init(self):
        ds = ToyRegDataset()
        ss = deepy.data.InverseDataset(ds)
        self.assertIsInstance(ss, deepy.data.dataset.InverseDataset)
        self.assertIsInstance(ss, torch.utils.data.Dataset)

    def test_len(self):
        ds = ToyRegDataset()
        ss = deepy.data.InverseDataset(ds)
        self.assertEqual(len(ss), len(ds))
    
    def test_getitem(self):
        ds = ToyRegDataset()
        ss = deepy.data.InverseDataset(ds)
        sample1, target1 = ds[0]
        sample2, target2 = ss[0]
        self.assertEqual(sample1, target2)
        self.assertEqual(sample2, target1)

        sample2, target2 = ss[10]
        self.assertEqual(sample2, 100)


class TestCaiMEFImageDataset(unittest.TestCase):

    def test_init(self):
        ds = deepy.data.vision.CaiMEImageDataset(
                 './tmp_data/', train=True, transform=None,
                 target_transform=None, transforms=None,
                 pre_load=False, pre_transform=None,
                 pre_target_transform=None,
                 pre_transforms=None, download=False)
        self.assertIsInstance(ds, torch.utils.data.Dataset)


if __name__ == "__main__":
    unittest.main()
