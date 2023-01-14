from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.transforms.base.kd_transform import KDTransform

from dataclasses import dataclass

@dataclass
class MultiViewConfig:
    n_views: int
    transform: callable


class MultiViewWrapper(KDWrapper):
    def __init__(self, configs, **kwargs):
        super().__init__(**kwargs)
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

    def getitem_x(self, idx, ctx=None):
        sample = self.dataset.getitem_x(idx)

        x = []
        counter = 0
        for config in self.transform_configs:
            for _ in range(config.n_views):
                view_ctx = {}
                if isinstance(config.transform, KDTransform):
                    view = config.transform(sample, ctx=view_ctx)
                else:
                    view = config.transform(sample)
                x.append(view)
                if ctx is not None:
                    ctx[f"view{counter}"] = view_ctx
                    counter += 1
        return x
