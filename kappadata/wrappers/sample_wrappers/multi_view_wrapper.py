from dataclasses import dataclass

import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.transforms import KDTransform, KDComposeTransform, KDStochasticTransform


@dataclass
class MultiViewConfig:
    n_views: int
    transform: callable


class MultiViewWrapper(KDWrapper):
    def __init__(self, dataset, configs, seed=None):
        super().__init__(dataset=dataset)
        assert isinstance(configs, list)
        # copy to not alter the original list
        configs = [*configs]
        # allow tuple/list/callable (e.g. MultiViewWrapper(configs=[ToTensor(), (2, ToTensor())])
        for i in range(len(configs)):
            if not isinstance(configs[i], MultiViewConfig):
                if isinstance(configs[i], (list, tuple)):
                    assert len(configs[i]) == 2
                    n_views, transform = configs[i]
                    configs[i] = MultiViewConfig(n_views=n_views, transform=transform)
                else:
                    configs[i] = MultiViewConfig(n_views=1, transform=configs[i])

        self.transform_configs = configs
        self.n_views = sum(config.n_views for config in configs)
        self.seed = seed

    def getitem_x(self, idx, ctx=None):
        sample = self.dataset.getitem_x(idx)
        x = []
        i = 0
        for config in self.transform_configs:
            # set rng of transforms
            if self.seed is not None:
                rng = np.random.default_rng(seed=self.seed + idx)
                if isinstance(config.transform, (KDComposeTransform, KDStochasticTransform)):
                    config.transform.set_rng(rng)
            # sample views
            for _ in range(config.n_views):
                view_ctx = {}
                if isinstance(config.transform, KDTransform):
                    view = config.transform(sample, ctx=view_ctx)
                else:
                    view = config.transform(sample)
                x.append(view)
                if ctx is not None:
                    ctx[f"view{i}"] = view_ctx
                    i += 1
        return x

    def worker_init_fn(self, rank, **kwargs):
        for config in self.transform_configs:
            if isinstance(config.transform, KDTransform):
                config.transform.worker_init_fn(rank, **kwargs)
