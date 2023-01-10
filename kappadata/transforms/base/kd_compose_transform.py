from .kd_random_apply_base import KDRandomApplyBase
from .kd_stochastic_transform import KDStochasticTransform
from .kd_transform import KDTransform
from kappadata.utils.is_deterministic_transform import has_stochastic_transform_with_seed, is_deterministic_transform

class KDComposeTransform(KDTransform):
    def __init__(self, transforms, check_consistent_seeds=True):
        self.transforms = transforms
        if check_consistent_seeds:
            if has_stochastic_transform_with_seed(transforms):
                assert is_deterministic_transform(transforms).all_kd_transforms_are_deterministic,\
                    f"transforms of type KDStochasticTransform within a KDComposeTransform should have: " \
                    f"1. seed is set for all KDStochasticTransforms or for none + " \
                    f"2. the seeds should be different to avoid patterns"


        # retrieve original_probs for rescaling apply probabilities
        self.original_probs = {
            i: transform.p
            for i, transform in enumerate(self.transforms)
            if isinstance(transform, KDRandomApplyBase)
        }

    def __call__(self, x, ctx=None):
        if ctx is None:
            ctx = {}
        for t in self.transforms:
            if isinstance(t, KDTransform):
                x = t(x, ctx)
            else:
                x = t(x)
        return x

    def reset_seed(self):
        for t in self.transforms:
            if isinstance(t, KDStochasticTransform):
                t.reset_seed()

    def scale_probs(self, scale):
        for i, original_prob in self.original_probs.items():
            self.transforms[i].p = original_prob * scale

    def reset_probs(self):
        for i, original_prob in self.original_probs.items():
            self.transforms.p = original_prob
