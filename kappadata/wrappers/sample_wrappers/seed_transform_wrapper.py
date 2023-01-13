import numpy as np
from kappadata.datasets.kd_wrapper import KDWrapper


# TODO test
class SeedTransformWrapper(KDWrapper):
    def __init__(self, seed, x_transform=None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(seed, int)
        self.x_transform = x_transform
        self.seed = seed

    def getitem_x(self, idx, ctx=None):
        raise RuntimeError(
            "a better way to implement this is to set the RNG directly...this avoids each composite/wrapper having to "
            "do something like set_seed(seed + i)"
        )
        if self.x_transform is None:
            return x
        self.x_transform.set_seed(self.seed + idx)
        return self.x_transform(self.dataset.getitem_x(idx, ctx=ctx))
