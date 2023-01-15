from functools import partial

from kappadata.datasets.kd_dataset import KDDataset
from .mode_wrapper import ModeWrapper


class TorchWrapper(KDDataset):
    def __init__(self, dataset, mode):
        super().__init__()
        self.dataset = dataset
        self.mode = mode

    def __getattr__(self, item):
        if item.startswith("getitem_"):
            item = item[len("getitem_"):]
            assert ModeWrapper.has_item(mode=self.mode, item=item)
            item_idx = ModeWrapper.get_item_index(mode=self.mode, item=item)
            return partial(self._getitem, item_idx=item_idx)
        return getattr(self.dataset, item)

    # noinspection PyUnusedLocal
    def _getitem(self, idx, ctx=None, item_idx=None):
        # item_idx has to default to None because ctx is not a kwarg so item_idx has to be the 3rd parameter otherwise
        # the ctx is passed into item_idx and the partial item_idx throws an error that item_idx is passed twice
        batch = self.dataset[idx]
        return batch[item_idx]

    def __len__(self):
        return len(self.dataset)
