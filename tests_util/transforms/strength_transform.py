from kappadata.transforms.base.kd_transform import KDTransform


class StrengthTransform(KDTransform):
    def __init__(self, strength):
        super().__init__()
        self.strength = self.og_strength = strength

    def _scale_strength(self, factor):
        self.strength = self.og_strength * factor

    def __call__(self, x, ctx=None):
        return self.strength