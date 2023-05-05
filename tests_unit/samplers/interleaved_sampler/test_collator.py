import unittest

import torch
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset

from kappadata.samplers.interleaved_sampler import InterleavedSampler, InterleavedSamplerConfig


class TestInterleavedSamplerCollator(unittest.TestCase):
    def _run(self, sampler, expected):
        loader_iter = iter(sampler.get_data_loader())
        for i in range(len(expected)):
            actual = next(loader_iter)
            if isinstance(actual, list):
                for j in range(len(actual)):
                    self.assertEqual(expected[i][j], actual[j].tolist())
            else:
                self.assertEqual(expected[i], actual.tolist())
        with self.assertRaises(StopIteration):
            next(loader_iter)

    def test_random_enu2sequential_nocollator(self):
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
                drop_last=False,
            ),
            expected=[
                # main
                [4, 1, 7, 5],
                [3, 9, 0, 8],
                # configs[0]
                [0, 1, 2, 3],
                [4],
                # main
                [6, 2],
            ],
        )

    def test_random_dataset(self):
        x = torch.randn(10, 3, 8, 8, generator=torch.Generator().manual_seed(43897))
        y = torch.randint(0, 10, size=(len(x),), generator=torch.Generator().manual_seed(4389787))
        self._run(
            sampler=InterleavedSampler(
                main_sampler=RandomSampler(TensorDataset(x, y), generator=torch.Generator().manual_seed(0)),
                batch_size=4,
                epochs=1,
                drop_last=False,
            ),
            expected=[
                (x[[4, 1, 7, 5]].tolist(), y[[4, 1, 7, 5]].tolist()),
                (x[[3, 9, 0, 8]].tolist(), y[[3, 9, 0, 8]].tolist()),
                (x[[6, 2]].tolist(), y[[6, 2]].tolist()),
            ],
        )

    def test_random_enu2sequential_maincollator(self):
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
                drop_last=False,
                main_collator=lambda data: torch.tensor([d + 1 for d in data])
            ),
            expected=[
                # main
                [5, 2, 8, 6],
                [4, 10, 1, 9],
                # configs[0]
                [0, 1, 2, 3],
                [4],
                # main
                [7, 3],
            ],
        )

    def test_random_enu2sequential_maincollator_interleavedcollator(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=RandomSampler(list(range(10)), generator=torch.Generator().manual_seed(0)),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_updates=2,
                        collator=lambda data: torch.tensor([d - 1 for d in data])
                    ),
                ],
                batch_size=4,
                epochs=1,
                drop_last=False,
                main_collator=lambda data: torch.tensor([d + 1 for d in data])
            ),
            expected=[
                # main
                [5, 2, 8, 6],
                [4, 10, 1, 9],
                # configs[0]
                [-1, 0, 1, 2],
                [3],
                # main
                [7, 3],
            ],
        )
