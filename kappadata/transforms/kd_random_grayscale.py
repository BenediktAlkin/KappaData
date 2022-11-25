import torchvision.transforms.functional as F

from .base.kd_stochastic_transform import KDStochasticTransform


class KDRandomGrayscale(KDStochasticTransform):
    def __init__(self, p=0.1, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def __call__(self, x, ctx=None):
        apply = self.rng.uniform() < self.p
        if ctx is not None:
            ctx["random_grayscale"] = apply

        num_output_channels = F.get_image_num_channels(x)
        if apply:
            return F.rgb_to_grayscale(x, num_output_channels=num_output_channels)
        return x