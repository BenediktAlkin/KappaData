import torch

from kappadata.datasets.kd_wrapper import KDWrapper


class LabelSmoothingWrapper(KDWrapper):
    def __init__(self, dataset, smoothing):
        super().__init__(dataset=dataset)
        assert isinstance(smoothing, (int, float)) and 0. < smoothing < 1.
        self.smoothing = smoothing

    def getitem_class(self, idx, ctx=None):
        y = self.dataset.getitem_class(idx, ctx)
        assert isinstance(y, int) or (torch.is_tensor(y) and y.ndim == 0)
        n_classes = self.dataset.getdim_class()
        off_value = self.smoothing / n_classes
        on_value = 1. - self.smoothing + off_value
        y_vector = torch.full(size=(n_classes,), fill_value=off_value)
        y_vector[y] = on_value
        return y_vector
