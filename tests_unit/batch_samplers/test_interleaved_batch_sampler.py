import unittest
from kappadata.batch_samplers.interleaved_batch_sampler import InterleavedBatchSampler


class TestInterleavedBatchSampler(unittest.TestCase):
    def test_check_last_idx_is_end_of_batch(self):
        sampler = [
            (False, 0)
        ]
        batch_sampler = InterleavedBatchSampler(sampler)
        batch_sampler_iter = iter(batch_sampler)
        with self.assertRaises(AssertionError):
            next(batch_sampler_iter)

    def test(self):
        sampler = [
            (False, 0),
            (True, 1),
            (False, 2),
            (False, 3),
            (True, 4),
            (False, 5),
            (False, 6),
            (False, 7),
            (False, 8),
            (True, 9),
        ]
        batch_sampler = InterleavedBatchSampler(sampler)
        expected_batches = [
            [0, 1],
            [2, 3, 4],
            [5, 6, 7, 8, 9],
        ]
        batch_sampler_iter = iter(batch_sampler)
        for i in range(len(expected_batches)):
            actual = next(batch_sampler_iter)
            self.assertEqual(expected_batches[i], actual)
        with self.assertRaises(StopIteration):
            next(batch_sampler_iter)