from kappadata.datasets.kd_dataset import KDDataset

class ClassDataset(KDDataset):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    def getitem_class(self, idx):
        return self.classes[idx]

    @property
    def n_classes(self):
        return max(self.classes)

    def __len__(self):
        return len(self.classes)