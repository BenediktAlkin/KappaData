import torch

from kappadata.datasets.kd_wrapper import KDWrapper


class LabelSmoothingWrapper(KDWrapper):
    def __init__(self, *args, smoothing, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(smoothing, (int, float)) and 0. < smoothing < 1.
        self.smoothing = smoothing

    def getitem_class(self, idx, ctx=None):
        y = self.dataset.getitem_class(idx, ctx)
        assert isinstance(y, int) or (torch.is_tensor(y) and y.ndim == 0)
        n_classes = self.dataset.n_classes
        off_value = self.smoothing / n_classes
        on_value = 1. - self.smoothing + off_value
        y_vector = torch.full(size=(n_classes,), fill_value=off_value)
        y_vector[y] = on_value
        return y_vector
