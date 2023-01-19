from kappadata.utils.is_deterministic_transform import has_stochastic_transform_with_seed, is_deterministic_transform
from .kd_stochastic_transform import KDStochasticTransform
from .kd_transform import KDTransform


class KDComposeTransform(KDTransform):
    def __init__(self, transforms, check_consistent_seeds=True):
        super().__init__()
        self.transforms = transforms
        # TODO seeds are deprecated
        if check_consistent_seeds:
            if has_stochastic_transform_with_seed(transforms):
                assert is_deterministic_transform(transforms).all_kd_transforms_are_deterministic, \
                    f"transforms of type KDStochasticTransform within a KDComposeTransform should have: " \
                    f"1. seed is set for all KDStochasticTransforms or for none + " \
                    f"2. the seeds should be different to avoid patterns"

    def __call__(self, x, ctx=None):
        if ctx is None:
            ctx = {}
        for t in self.transforms:
            if isinstance(t, KDTransform):
                x = t(x, ctx)
            else:
                x = t(x)
        return x

    # TODO reset_seed is deprecated
    def reset_seed(self):
        for t in self.transforms:
            if isinstance(t, KDStochasticTransform):
                t.reset_seed()

    def set_rng(self, rng):
        for t in self.transforms:
            if isinstance(t, KDStochasticTransform):
                t.set_rng(rng)

    def _scale_strength(self, factor):
        for t in self.transforms:
            if isinstance(t, KDTransform):
                t.scale_strength(factor)
