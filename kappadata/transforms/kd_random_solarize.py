import torchvision.transforms.functional as F

from .base.kd_random_apply_base import KDRandomApplyBase


class KDRandomSolarize(KDRandomApplyBase):
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def forward(self, x, ctx):
        # if ctx is not None:
        #     ctx["random_solarize"] = True
        return F.solarize(x, self.threshold)
