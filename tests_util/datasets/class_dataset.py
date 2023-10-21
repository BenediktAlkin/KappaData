from kappadata.datasets.kd_dataset import KDDataset


class ClassDataset(KDDataset):
    def __init__(self, classes, n_classes=None, class_names=None):
        super().__init__()
        self.classes = classes
        self._n_classes = n_classes
        if class_names is not None:
            assert len(class_names) == self.getdim_class()
            self._class_names = class_names

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.classes[idx]

    def getshape_class(self):
        n_classes = self._n_classes or max(self.classes) + 1
        if n_classes == 2:
            n_classes = 1
        return n_classes,

    @property
    def class_names(self):
        assert self._class_names is not None
        return self._class_names

    def __len__(self):
        return len(self.classes)

    def getall_class(self):
        return [c for c in self.classes]
