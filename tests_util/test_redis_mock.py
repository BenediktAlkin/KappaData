import unittest

import redis
import torch

from tests_util.redis_mock import RedisMock


class TestRedisMock(unittest.TestCase):
    def test_encode(self):
        kwargs = dict(host="localhost", port=6379)
        og = redis.Redis(**kwargs)
        mock = RedisMock(**kwargs)

        sources = [5, 10., "test", b"102", torch.tensor([1., 2.])]
        for i, source in enumerate(sources):
            og.set(i, source)
            mock.set(i, source)
            self.assertTrue(og.exists(i))
            self.assertTrue(mock.exists(i))
            og_actual = og.get(i)
            mock_actual = og.get(i)
            self.assertEqual(og_actual, mock_actual)
