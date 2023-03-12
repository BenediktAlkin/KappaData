import unittest

from kappadata.samplers.interleaved_sampler import InterleavedSampler, InterleavedSamplerConfig
from kappadata.batch_samplers.interleaved_batch_sampler import InterleavedBatchSampler
from torch.utils.data import SequentialSampler, RandomSampler


class TestInterleavedSamplers(unittest.TestCase):
    def _run(self, sampler, expected):
        batch_sampler = InterleavedBatchSampler(sampler)
        batch_sampler_iter = iter(batch_sampler)
        for i in range(len(expected)):
            actual = next(batch_sampler_iter)
            self.assertEqual(expected[i], actual)
        with self.assertRaises(StopIteration):
            next(batch_sampler_iter)

    def test_sequential_nodroplast_enu1sequential(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=SequentialSampler(list(range(10))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_updates=1,
                    ),
                ],
                batch_size=4,
                drop_last=False,
                epochs=1,
            ),
            expected=[
                # main
                [0, 1, 2, 3],
                # configs[0]
                [10, 11, 12, 13],
                [14],
                # main
                [4, 5, 6, 7],
                # configs[0]
                [10, 11, 12, 13],
                [14],
                # main
                [8, 9],
                # configs[0](last batch is counted as an update as drop_last=False)
                [10, 11, 12, 13],
                [14],
            ],
        )