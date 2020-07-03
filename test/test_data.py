import unittest
import deepy.data

import torch


class ToyDataset(torch.util.data.Dataset):
    def __init__(self):
        super(ToyDataset, self).__init__()
    
    def __getitem__(self, index):
        sample = index
        target = index**2
        return sample, target
    
    def __len__(self):
        return 100


class TestClassDataset(unittest.TestCase):
    pass


class TestRegDataset(unittest.TestCase):
    pass


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
        ds = ToyDataset()
        ss = deepy.data.SelfSupervisedDataset(ds)
        self.assertIsInstance(ss, deepy.data.SelfSupervisedDataset)
        self.assertIsInstance(ss, torch.util.data.Dataset)

    def test_len(self):
        ds = ToyDataset()
        ss = deepy.data.SelfSupervisedDataset(ds)
        self.assertEqual(len(ss), len(ds))
    
    def test_getitem(self):
        ds = ToyDataset()
        ss = deepy.data.SelfSupervisedDataset(ds)
        sample1, *_ = ds[0]
        sample2, target2 = ss[0]
        self.assertEqual(sample2, target2)

        sample2, target2 = ss[10]
        self.assertEqual(sample2, 10)


class TestInverseDataset(unittest.TestCase):

    def test_init(self):
        ds = ToyDataset()
        ss = deepy.data.InverseDataset(ds)
        self.assertIsInstance(ss, deepy.data.InverseDataset)
        self.assertIsInstance(ss, torch.util.data.Dataset)

    def test_len(self):
        ds = ToyDataset()
        ss = deepy.data.InverseDataset(ds)
        self.assertEqual(len(ss), len(ds))
    
    def test_getitem(self):
        ds = ToyDataset()
        ss = deepy.data.InverseDataset(ds)
        sample1, target1 = ds[0]
        sample2, target2 = ss[0]
        self.assertEqual(sample1, target2)
        self.assertEqual(sample2, target1)

        sample2, target2 = ss[10]
        self.assertEqual(sample2, 100)


if __name__ == "__main__":
    unittest.main()
