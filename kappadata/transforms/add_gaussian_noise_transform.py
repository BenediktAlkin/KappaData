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
        self.strength = self.og_strength = 1.

    def _scale_strength(self, factor):
        self.strength = factor

    def __call__(self, x, ctx=None):
        magnitude = self.magnitude_sampler.sample(self.rng)
        noise = torch.from_numpy(self.rng.normal(scale=magnitude, size=x.shape)).float()
        return x + noise
