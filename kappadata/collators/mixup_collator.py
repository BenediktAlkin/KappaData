import torch
import numpy as np
from torch.nn.functional import one_hot

class MixupCollator:
    def __init__(self, alpha, p=1., n_classes=None, seed=None):
        assert isinstance(alpha, (int, float)) and 0. < alpha
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        assert n_classes is None or (isinstance(n_classes, int) and n_classes > 1)
        self.alpha = alpha
        self.p = p
        self.n_classes = n_classes
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed=seed)

    def __call__(self, batch):
        assert len(batch) == 2
        x, y = batch
        batch_size = len(x)
        apply = torch.rand(size=(batch_size,), generator=self.rng) < self.p
        lamb = torch.from_numpy(self.np_rng.beta(self.alpha, self.alpha, size=batch_size)).type(torch.float32)
        idx2 = torch.from_numpy(self.np_rng.permutation(batch_size)).type(torch.long)

        if y.ndim == 1:
            y = one_hot(y, num_classes=self.n_classes).type(torch.float32)

        # add dimensions for broadcasting
        apply_x = apply.view(-1, *[1] * (x.ndim - 1))
        apply_y = apply.view(-1, 1)
        lamb_x = lamb.view(-1, *[1] * (x.ndim - 1))
        lamb_y = lamb.view(-1, 1)

        mixed_x = torch.where(apply_x, lamb_x * x + (1. - lamb_x) * x[idx2], x)
        mixed_y = torch.where(apply_y, lamb_y * y + (1. - lamb_y) * y[idx2], y)
        return mixed_x, mixed_y