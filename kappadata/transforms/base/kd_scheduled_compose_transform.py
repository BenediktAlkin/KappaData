from .kd_compose_transform import KDComposeTransform
from .kd_random_apply_base import KDRandomApplyBase

class KDScheduledComposeTransform(KDComposeTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_probs = {
            i: transform.p
            for i, transform in enumerate(self.transforms)
            if isinstance(transform, KDRandomApplyBase)
        }

    def scale_probs(self, scale):
        for i, original_prob in self.original_probs.items():
            self.transforms[i].p = original_prob * scale

    def reset_probs(self):
        for i, original_prob in self.original_probs.items():
            self.transforms.p = original_prob