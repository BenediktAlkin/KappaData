class KDTransform:
    def __init__(self, ctx_prefix=None):
        self.ctx_prefix = ctx_prefix or type(self).__name__
        # sanity check to avoid accidentally overwriting base method
        assert type(self).scale_strength == KDTransform.scale_strength

    def __call__(self, x, ctx=None):
        raise NotImplementedError

    def worker_init_fn(self, rank, **kwargs):
        pass

    def scale_strength(self, factor):
        assert 0. <= factor <= 1.
        self._scale_strength(factor)

    def _scale_strength(self, factor):
        pass

    @classmethod
    def supports_scale_strength(cls):
        return cls._scale_strength != KDTransform._scale_strength
