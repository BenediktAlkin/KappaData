from torchvision.transforms.functional import hflip

from .kd_random_apply import KDRandomApply


class KDRandomHorizontalFlip(KDRandomApply):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(p=p, **kwargs)

    def forward(self, x, ctx):
        if ctx is not None:
            ctx["random_hflip"] = True
        return hflip(x)
