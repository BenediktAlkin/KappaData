import math
from collections import defaultdict

import numpy as np
import torch

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.global_rng import GlobalRng


class ClassGroupsWrapper(KDWrapper):
    def __init__(self, dataset, classes_per_group, shuffle=False, seed=None):
        super().__init__(dataset=dataset)
        self.classes_per_group = classes_per_group
        rng = GlobalRng() if seed is None else np.random.default_rng(seed=seed)
        classes = dataset.getall_class()
        if torch.is_tensor(classes):
            classes = classes.tolist()
        assert isinstance(classes, (tuple, list))

        num_clsgroups = math.ceil(dataset.getdim_class() / self.classes_per_group)
        self.cls_to_clsgroup = np.arange(num_clsgroups).repeat(self.classes_per_group)
        if shuffle:
            self.cls_to_clsgroup = rng.permuted(self.cls_to_clsgroup)

        self.idx_within_class = []
        counter = defaultdict(int)
        for cls in classes:
            self.idx_within_class.append(counter[cls])
            counter[cls] += 1

    def _map_cls(self, idx, cls):
        cls_group = self.cls_to_clsgroup[cls]
        idx_within_cls_group = self.idx_within_class[idx] % self.classes_per_group
        cls = cls_group * self.classes_per_group + idx_within_cls_group
        return cls

    def getitem_class(self, idx, ctx=None):
        cls = self.dataset.getitem_class(idx, ctx=ctx)
        return self._map_cls(idx=idx, cls=cls)

    def getall_class(self):
        classes = self.dataset.getall_class()
        return [self._map_cls(idx=idx, cls=cls) for idx, cls in enumerate(classes)]
