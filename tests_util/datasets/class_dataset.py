from kappadata.datasets.kd_dataset import KDDataset


class ClassDataset(KDDataset):
    def __init__(self, classes, n_classes=None, class_names=None):
        super().__init__()
        self.classes = classes
        self._n_classes = n_classes
        if class_names is not None:
            assert len(class_names) == self.n_classes
            self._class_names = class_names

    def getitem_class(self, idx, _=None):
        return self.classes[idx]

    @property
    def n_classes(self):
        return self._n_classes or max(self.classes) + 1

    @property
    def class_names(self):
        assert self._class_names is not None
        return self._class_names

    def __len__(self):
        return len(self.classes)

    def getall_class(self):
        return [c for c in self.classes]