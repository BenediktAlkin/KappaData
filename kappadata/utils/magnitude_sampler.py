import numpy as np


class MagnitudeSampler:
    def __init__(
            self,
            magnitude: float,
            magnitude_std: float = 0.,
            magnitude_min: float = 0.,
            magnitude_max: float = 1.,
    ):
        assert isinstance(magnitude, (int, float)) and 0 <= magnitude
        assert isinstance(magnitude_std, (int, float)) and 0. <= magnitude_std
        assert isinstance(magnitude_min, (int, float)) and 0. <= magnitude_min <= magnitude
        assert isinstance(magnitude_max, (int, float)) and magnitude <= magnitude_max
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std
        self.magnitude_min = magnitude_min
        self.magnitude_max = magnitude_max
        if magnitude_std == 0.:
            self.sample = self._sample_const
        elif magnitude_std == float("inf"):
            self.sample = self._sample_uniform
        else:
            self.sample = self._sample_normal

    def _sample_const(self, _):
        return self.magnitude

    def _sample_uniform(self, rng):
        return rng.uniform(self.magnitude_min, self.magnitude)

    def _sample_normal(self, rng):
        sampled = self.magnitude + rng.normal(0, self.magnitude_std / 10)
        # convert to python float to be consistent with other sampling value dtypes (np.clip converts to np.float64)
        return float(np.clip(sampled, self.magnitude_min, self.magnitude_max))
