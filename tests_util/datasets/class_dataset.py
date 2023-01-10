from kappadata.datasets.kd_dataset import KDDataset


class ClassDataset(KDDataset):
    def __init__(self, classes, n_classes=None):
        super().__init__()
        self.classes = classes
        self._n_classes = n_classes

    def getitem_class(self, idx, _=None):
        return self.classes[idx]

    @property
    def n_classes(self):
        return self._n_classes or max(self.classes) + 1

    def __len__(self):
        return len(self.classes)
