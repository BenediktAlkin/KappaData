from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform


class KDRandomApplyBase(KDStochasticTransform):
    def __init__(self, p, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def __call__(self, x, ctx=None):
        apply = self.rng.uniform() < self.p
        if apply:
            return self.forward(x, ctx)
        return x

    def forward(self, x, ctx):
        raise NotImplementedError
