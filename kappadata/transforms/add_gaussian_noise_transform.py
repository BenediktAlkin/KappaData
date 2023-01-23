import torch

from kappadata.utils.magnitude_sampler import MagnitudeSampler
from .base.kd_stochastic_transform import KDStochasticTransform


class AddGaussianNoiseTransform(KDStochasticTransform):
    def __init__(
            self,
            magnitude: float = 1.,
            magnitude_std: float = float("inf"),
            magnitude_min: float = 0.,
            magnitude_max: float = 1.,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.magnitude_sampler = MagnitudeSampler(
            magnitude=magnitude,
            magnitude_std=magnitude_std,
            magnitude_min=magnitude_min,
            magnitude_max=magnitude_max,
        )
        self.ctx_key = f"{self.ctx_prefix}.magnitude"

    def _scale_strength(self, factor):
        self.magnitude_sampler.scale_strength(factor)

    def __call__(self, x, ctx=None):
        magnitude = self.magnitude_sampler.sample(self.rng)
        noise = torch.from_numpy(self.rng.normal(scale=magnitude, size=x.shape)).float()
        if ctx is not None:
            ctx[self.ctx_key] = magnitude
        return x + noise
