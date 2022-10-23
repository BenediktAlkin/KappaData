import numpy as np

from kappadata.datasets.kd_subset import KDSubset


class ClassFilterWrapper(KDSubset):
    def __init__(self, dataset, valid_classes=None, invalid_classes=None):
        assert (valid_classes is None) ^ (invalid_classes is None)
        assert valid_classes is None or isinstance(valid_classes, list)
        assert invalid_classes is None or isinstance(invalid_classes, list)
        self.valid_classes = set(valid_classes) if valid_classes is not None else None
        self.invalid_classes = set(invalid_classes) if invalid_classes is not None else None

        # use numpy for better performance
        # NOTE: np.isin requires list (not set)
        all_indices = np.arange(len(dataset))
        classes = np.array([dataset.getitem_class(i) for i in all_indices])
        if self.valid_classes is not None:
            indices = all_indices[np.isin(classes, list(self.valid_classes))]
        else:
            indices = all_indices[~np.isin(classes, list(self.invalid_classes))]
        super().__init__(dataset=dataset, indices=indices)
