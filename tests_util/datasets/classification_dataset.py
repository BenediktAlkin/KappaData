from kappadata.datasets.kd_dataset import KDDataset


class ClassificationDataset(KDDataset):
    def __init__(self, x, classes):
        super().__init__()
        assert len(x) == len(classes)
        self.x = x
        self.classes = classes

    def getitem_x(self, idx, _=None):
        return self.x[idx].clone()

    def getitem_class(self, idx, _=None):
        return self.classes[idx]

    @property
    def n_classes(self):
        return max(self.classes) + 1

    def __len__(self):
        return len(self.classes)
