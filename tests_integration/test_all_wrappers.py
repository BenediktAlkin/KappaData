import unittest

from kappadata.wrappers.dataset_wrappers.percent_filter_wrapper import PercentFilterWrapper
from kappadata.wrappers.dataset_wrappers.repeat_wrapper import RepeatWrapper
from kappadata.wrappers.dataset_wrappers.shuffle_wrapper import ShuffleWrapper
from kappadata.wrappers.sample_wrappers.mixup_wrapper import MixupWrapper
from tests_util.index_dataset import IndexDataset


class TestAllWrappers(unittest.TestCase):
    def test(self):
        root_ds = IndexDataset(size=10)
        wrapper1 = PercentFilterWrapper(root_ds, from_percent=0.3)
        wrapper2 = RepeatWrapper(wrapper1, repetitions=2)
        wrapper3 = MixupWrapper(wrapper2, alpha=1., p=1.)
        wrapper4 = ShuffleWrapper(wrapper3, seed=2)
        wrapper5 = PercentFilterWrapper(wrapper4, to_percent=0.5)

        self.assertEqual([wrapper5, wrapper4, wrapper3, wrapper2, wrapper1], wrapper5.all_wrappers)
        self.assertEqual([wrapper4, wrapper3, wrapper2, wrapper1], wrapper4.all_wrappers)
        self.assertEqual([wrapper3, wrapper2, wrapper1], wrapper3.all_wrappers)
        self.assertEqual([wrapper2, wrapper1], wrapper2.all_wrappers)
        self.assertEqual([wrapper1], wrapper1.all_wrappers)
