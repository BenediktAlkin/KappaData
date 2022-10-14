import torch
import unittest
from unittest.mock import patch
import redis
from kappadata.caching.redis_dataset import RedisDataset
from tests_mock.redis_mock import RedisMock
from torch.utils.data import Dataset

class TestRedisDataset(unittest.TestCase):
    @patch("kappadata.caching.redis_dataset.redis")
    def test_encode_decode(self, redis_module):
        class TestDataset(Dataset):
            def __getitem__(self, item):
                float_tensor = torch.tensor([3., 4.], dtype=torch.float32)
                long_tensor = torch.tensor([5, 6], dtype=torch.long)
                return 1, 2., "test", True, b"12", float_tensor, long_tensor

            def __len__(self):
                return 1

        redis_module.Redis = RedisMock
        ds = TestDataset()
        redis_ds = RedisDataset(dataset=ds, host="localhost", port=6379)

        def test_equal(expected_sample, actual_sample):
            for expected, actual in zip(expected_sample, actual_sample):
                if torch.is_tensor(expected):
                    self.assertEqual(expected.dtype, actual.dtype)
                    self.assertTrue(torch.all(expected == actual))
                else:
                    self.assertEqual(expected, actual)


        test_equal(ds[0], redis_ds[0])
        test_equal(ds[0], redis_ds[0])
