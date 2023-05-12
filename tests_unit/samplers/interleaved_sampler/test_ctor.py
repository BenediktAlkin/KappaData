import unittest

from torch.utils.data import SequentialSampler

from kappadata.samplers.interleaved_sampler import InterleavedSampler, InterleavedSamplerConfig


class TestInterleavedSamplerCtor(unittest.TestCase):
    def test_invalid_batch_size(self):
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size=0,
                epochs=1,
            )
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size=4.,
                epochs=1,
            )
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size="4",
                epochs=1,
            )

    def test_batch_size_too_high(self):
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(2))),
                batch_size=4,
                epochs=1,
            )

    def test_invalid_duration(self):
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size=4,
            )
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size=4,
                epochs=3.2,
            )
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size=4,
                epochs=-3,
            )

    def test_no_config_duration(self):
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                configs=[
                    InterleavedSamplerConfig(
                        sampler=SequentialSampler(list(range(5))),
                    ),
                ],
                batch_size=4,
            )

    def test_multiple_durations(self):
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size=4,
                epochs=1,
                updates=1,
            )
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size=4,
                epochs=1,
                samples=1,
            )
        with self.assertRaises(AssertionError):
            InterleavedSampler(
                main_sampler=SequentialSampler(list(range(4))),
                batch_size=4,
                updates=1,
                samples=1,
            )
