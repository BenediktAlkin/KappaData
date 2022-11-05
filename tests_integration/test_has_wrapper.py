import unittest

from kappadata.wrappers.dataset_wrappers.percent_filter_wrapper import PercentFilterWrapper
from kappadata.wrappers.dataset_wrappers.repeat_wrapper import RepeatWrapper
from kappadata.wrappers.dataset_wrappers.shuffle_wrapper import ShuffleWrapper
from kappadata.wrappers.sample_wrappers.mixup_wrapper import MixupWrapper
from tests_util.index_dataset import IndexDataset


class TestHasWrapper(unittest.TestCase):
    def test(self):
        root_ds = IndexDataset(size=10)
        wrapper1 = PercentFilterWrapper(root_ds, from_percent=0.3)
        wrapper2 = RepeatWrapper(wrapper1, repetitions=2)
        wrapper3 = MixupWrapper(wrapper2, alpha=1., p=1.)
        wrapper4 = ShuffleWrapper(wrapper3, seed=2)
        wrapper5 = PercentFilterWrapper(wrapper4, to_percent=0.5)

        self.assertTrue(wrapper5.has_wrapper(wrapper1))
        self.assertTrue(wrapper5.has_wrapper(wrapper2))
        self.assertTrue(wrapper5.has_wrapper(wrapper3))
        self.assertTrue(wrapper5.has_wrapper(wrapper4))
        self.assertTrue(wrapper5.has_wrapper(wrapper5))

        self.assertTrue(wrapper4.has_wrapper(wrapper1))
        self.assertTrue(wrapper4.has_wrapper(wrapper2))
        self.assertTrue(wrapper4.has_wrapper(wrapper3))
        self.assertTrue(wrapper4.has_wrapper(wrapper4))
        self.assertFalse(wrapper4.has_wrapper(wrapper5))

        self.assertTrue(wrapper3.has_wrapper(wrapper1))
        self.assertTrue(wrapper3.has_wrapper(wrapper2))
        self.assertTrue(wrapper3.has_wrapper(wrapper3))
        self.assertFalse(wrapper3.has_wrapper(wrapper4))
        self.assertFalse(wrapper3.has_wrapper(wrapper5))

        self.assertTrue(wrapper2.has_wrapper(wrapper1))
        self.assertTrue(wrapper2.has_wrapper(wrapper2))
        self.assertFalse(wrapper2.has_wrapper(wrapper3))
        self.assertFalse(wrapper2.has_wrapper(wrapper4))
        self.assertFalse(wrapper2.has_wrapper(wrapper5))

        self.assertTrue(wrapper1.has_wrapper(wrapper1))
        self.assertFalse(wrapper1.has_wrapper(wrapper2))
        self.assertFalse(wrapper1.has_wrapper(wrapper3))
        self.assertFalse(wrapper1.has_wrapper(wrapper4))
        self.assertFalse(wrapper1.has_wrapper(wrapper5))
