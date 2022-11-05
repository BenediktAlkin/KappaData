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

        self.assertTrue(wrapper5.has_wrapper_type(PercentFilterWrapper))
        self.assertTrue(wrapper5.has_wrapper_type(RepeatWrapper))
        self.assertTrue(wrapper5.has_wrapper_type(MixupWrapper))
        self.assertTrue(wrapper5.has_wrapper_type(ShuffleWrapper))

        self.assertTrue(wrapper4.has_wrapper_type(PercentFilterWrapper))
        self.assertTrue(wrapper4.has_wrapper_type(RepeatWrapper))
        self.assertTrue(wrapper4.has_wrapper_type(MixupWrapper))
        self.assertTrue(wrapper4.has_wrapper_type(ShuffleWrapper))

        self.assertTrue(wrapper3.has_wrapper_type(PercentFilterWrapper))
        self.assertTrue(wrapper3.has_wrapper_type(RepeatWrapper))
        self.assertTrue(wrapper3.has_wrapper_type(MixupWrapper))
        self.assertFalse(wrapper3.has_wrapper_type(ShuffleWrapper))

        self.assertTrue(wrapper2.has_wrapper_type(PercentFilterWrapper))
        self.assertTrue(wrapper2.has_wrapper_type(RepeatWrapper))
        self.assertFalse(wrapper2.has_wrapper_type(MixupWrapper))
        self.assertFalse(wrapper2.has_wrapper_type(ShuffleWrapper))

        self.assertTrue(wrapper1.has_wrapper_type(PercentFilterWrapper))
        self.assertFalse(wrapper1.has_wrapper_type(ShuffleWrapper))
        self.assertFalse(wrapper1.has_wrapper_type(MixupWrapper))
        self.assertFalse(wrapper1.has_wrapper_type(RepeatWrapper))
