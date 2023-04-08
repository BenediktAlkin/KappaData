import unittest

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from kappadata.samplers.interleaved_sampler import InterleavedSampler, InterleavedSamplerConfig


class TestInterleavedSamplerSampler(unittest.TestCase):
    def _run(self, sampler, expected):
        actual = [i for (_, i) in sampler]
        self.assertEqual(expected, actual)

    def test_sequential_nodroplast_ene1sequential(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=SequentialSampler(list(range(10))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_epochs=1,
                    ),
                ],
                batch_size=4,
                drop_last=False,
                epochs=2,
            ),
            expected=[
                # main
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9,
                # configs[0]
                10, 11, 12, 13, 14,
            ],
        )

    def test_sequential_nodroplast_ene2sequential(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=SequentialSampler(list(range(10))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_epochs=2,
                    ),
                ],
                batch_size=4,
                drop_last=False,
                epochs=2,
            ),
            expected=[
                # main
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9,
                # main
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9,
                # configs[0]
                10, 11, 12, 13, 14,
            ],
        )

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
                # main
                0, 1, 2, 3,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                4, 5, 6, 7,
                # configs[0]
                10, 11, 12, 13, 14,
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
                # main
                0, 1, 2, 3,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                4, 5, 6, 7,
                # configs[0]
                10, 11, 12, 13, 14,
                # configs[1]
                15, 16,
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
                drop_last=False,
            ),
            expected=[
                # main
                0, 1, 2, 3, 4, 5, 6, 7,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                8, 9,
            ],
        )

    def test_sequential_nodroplast_ens8sequential(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=SequentialSampler(list(range(10))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_samples=8,
                    ),
                ],
                batch_size=4,
                drop_last=False,
                epochs=2,
            ),
            expected=[
                # main
                0, 1, 2, 3,
                4, 5, 6, 7,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                8, 9,
                0, 1, 2, 3,
                4, 5, 6, 7,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                8, 9,
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
                drop_last=False,
            ),
            expected=[
                # main
                4, 1, 7, 5, 3, 9, 0, 8,
                # configs[0]
                10, 11, 12, 13, 14,
                # main
                6, 2,
            ],
        )

    def test_distsequential_enu1distsequential_rank0of2(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=DistributedSampler(list(range(10)), shuffle=False, num_replicas=2, rank=0),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=DistributedSampler(list(range(5)), shuffle=False, num_replicas=2, rank=0),
                        every_n_updates=1,
                    ),
                ],
                batch_size=4,
                epochs=1,
                drop_last=False,
            ),
            expected=[
                # main
                0, 2, 4, 6,
                # configs[0]
                10, 12, 14,
                # main
                8,
                # configs[0]
                10, 12, 14,
            ],
        )

    def test_distsequential_enu1distsequential_rank1of2(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=DistributedSampler(list(range(10)), shuffle=False, num_replicas=2, rank=1),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=DistributedSampler(list(range(5)), shuffle=False, num_replicas=2, rank=1),
                        every_n_updates=1,
                    ),
                ],
                batch_size=4,
                epochs=1,
                drop_last=False,
            ),
            expected=[
                # main
                1, 3, 5, 7,
                # configs[0]
                11, 13, 10,
                # main
                9,
                # configs[0]
                11, 13, 10,
            ],
        )




    def test_sequential_droplast_nofullepoch(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=SequentialSampler(list(range(100))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_samples=8,
                    ),
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_epochs=1,
                    ),
                ],
                batch_size=4,
                drop_last=True,
                samples=16,
            ),
            expected=[
                # main
                0, 1, 2, 3,
                4, 5, 6, 7,
                # config[0]
                100, 101, 102, 103, 104,
                # main
                8, 9, 10, 11,
                12, 13, 14, 15,
                # config[0]
                100, 101, 102, 103, 104,
            ],
        )


    def test_sequential_droplast_nointervalonsampleend(self):
        self._run(
            sampler=InterleavedSampler(
                main_sampler=SequentialSampler(list(range(100))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                        every_n_samples=6,
                    ),
                ],
                batch_size=4,
                drop_last=True,
                samples=16,
            ),
            expected=[
                # main
                0, 1, 2, 3,
                4, 5, 6, 7,
                # config[0]
                100, 101, 102, 103, 104,
                # main
                8, 9, 10, 11,
                # config[0]
                100, 101, 102, 103, 104,
                # main
                12, 13, 14, 15,
            ],
        )