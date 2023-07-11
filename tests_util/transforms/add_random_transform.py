from kappadata.transforms import KDStochasticTransform


class AddRandomTransform(KDStochasticTransform):
    def __call__(self, x, ctx=None):
        return [x, self.rng.random()]
