from kappadata.transforms import KDStochasticTransform


class ReplaceWithRandomTransform(KDStochasticTransform):
    def __call__(self, x, ctx=None):
        return self.rng.random()
