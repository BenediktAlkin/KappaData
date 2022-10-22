from kappadata.datasets.kd_subset import KDSubset


class ClassFilterWrapper(KDSubset):
    def __init__(self, dataset, valid_classes=None, invalid_classes=None):
        assert (valid_classes is None) ^ (invalid_classes is None)
        assert valid_classes is None or isinstance(valid_classes, list)
        assert invalid_classes is None or isinstance(invalid_classes, list)
        self.valid_classes = set(valid_classes) if valid_classes is not None else None
        self.invalid_classes = set(invalid_classes) if invalid_classes is not None else None

        indices = [i for i in range(len(dataset)) if self._is_valid_class(dataset.getitem_class(i))]
        super().__init__(dataset=dataset, indices=indices)

    def _is_valid_class(self, cls):
        if self.valid_classes is not None:
            return cls in self.valid_classes
        else:
            return cls not in self.invalid_classes
