import torch
import unittest

from kappadata.samplers.interleaved_sampler import InterleavedSampler, InterleavedSamplerConfig
from torch.utils.data import RandomSampler, SequentialSampler

class TestInterleavedSampler(unittest.TestCase):
    def _run(self, sampler, expected):
        actual = [i for (_, i) in sampler]
        self.assertEqual(expected, actual)

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
                0, 1, 2, 3,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                4, 5, 6, 7,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                8, 9,
                # configs[0] (last batch is counted as an update as drop_last=False)
                10, 11, 12, 13, 14,
            ],
        )

    def test_sequential_droplast_enu1sequential(self):
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
                drop_last=True,
                epochs=1,
            ),
            expected=[
                0, 1, 2, 3,  # main
                10, 11, 12, 13, 14,  # configs[0]
                4, 5, 6, 7,  # main
                10, 11, 12, 13, 14,  # configs[0]
            ],
        )

    def test_sequential_droplast_enu1sequential_enu2sequential(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=SequentialSampler(list(range(10))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_updates=1,
                    ),
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(2))),
                        every_n_updates=2,
                    ),
                ],
                batch_size=4,
                drop_last=True,
                epochs=1,
            ),
            expected=[
                0, 1, 2, 3,  # main
                10, 11, 12, 13, 14,  # configs[0]
                4, 5, 6, 7,  # main
                10, 11, 12, 13, 14,  # configs[0]
                15, 16,  # configs[1]
            ],
        )

    def test_sequential_enu2sequential(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=SequentialSampler(list(range(10))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_updates=2,
                    ),
                ],
                batch_size=4,
                epochs=1,
            ),
            expected=[
                0, 1, 2, 3, 4, 5, 6, 7,  # main
                10, 11, 12, 13, 14,  # configs[0]
                8, 9,  # main
            ],
        )

    def test_random_enu2sequential(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=RandomSampler(list(range(10)), generator=torch.Generator().manual_seed(0)),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_updates=2,
                    ),
                ],
                batch_size=4,
                epochs=1,
            ),
            expected=[
                4, 1, 7, 5, 3, 9, 0, 8,  # main
                10, 11, 12, 13, 14,  # configs[0]
                6, 2,  # main
            ],
        )