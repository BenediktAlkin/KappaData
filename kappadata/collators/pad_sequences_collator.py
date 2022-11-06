import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .base.kd_collator import KDCollator
from torch.utils.data import default_collate


class PadSequencesCollator(KDCollator):
    @property
    def default_collate_mode(self):
        return None

    def collate(self, batch, _, ctx=None):
        if isinstance(batch[0], tuple):
            result = []
            for i in range(len(batch[0])):
                first_item = batch[0][i]
                items = [b[i] for b in batch]
                if torch.is_tensor(first_item) and first_item.ndim > 0:
                    result.append(pad_sequence([b[i] for b in batch], batch_first=True))
                else:
                    result.append(default_collate(items))
            return tuple(result)
        return pad_sequence(batch, batch_first=True)