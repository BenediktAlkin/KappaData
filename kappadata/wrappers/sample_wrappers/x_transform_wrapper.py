import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.transforms import KDComposeTransform, KDStochasticTransform


class XTransformWrapper(KDWrapper):
    def __init__(self, dataset, transform, seed=None):
        super().__init__(dataset=dataset)
        self.transform = transform
        self.seed = seed

    def getitem_x(self, idx, ctx=None):
        x = self.dataset.getitem_x(idx, ctx=ctx)
        if self.seed is not None:
            rng = np.random.default_rng(seed=self.seed + idx)
            if isinstance(self.transform, (KDComposeTransform, KDStochasticTransform)):
                self.transform.set_rng(rng)
        return self.transform(x)