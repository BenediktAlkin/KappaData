import unittest

from torch.utils.data import SequentialSampler

from kappadata.samplers.interleaved_sampler import InterleavedSampler


class TestInterleavedSampler(unittest.TestCase):
    def test_batch_size_too_high(self):
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(2))),
                batch_size=4,
                drop_last=False,
                epochs=1,
            )
