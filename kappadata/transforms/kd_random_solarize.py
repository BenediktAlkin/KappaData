import torchvision.transforms.functional as F

from .base.kd_random_apply_base import KDRandomApplyBase


class KDRandomSolarize(KDRandomApplyBase):
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = self.og_threshold = threshold

    def _scale_strength(self, factor):
        # PIL -> threshold >= 256 -> no augmentation
        # tensor -> threshold >= 1. -> no augmentation
        if isinstance(self.og_threshold, int):
            # PIL
            self.threshold = int(256 - (256 - self.og_threshold) * factor)
        elif isinstance(self.og_threshold, float):
            # tensor
            self.threshold = 1. - (1. - self.og_threshold) * factor
        else:
            raise NotImplementedError

    def forward(self, x, ctx):
        # if ctx is not None:
        #     ctx["random_solarize"] = True
        return F.solarize(x, self.threshold)
