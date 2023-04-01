import unittest

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from kappadata.samplers.interleaved_sampler import InterleavedSampler, InterleavedSamplerConfig


class TestInterleavedSampler(unittest.TestCase):
    def test_batch_size_too_high(self):
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(2))),
                batch_size=4,
                drop_last=False,
                epochs=1,
            )
