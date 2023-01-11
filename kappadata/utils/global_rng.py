import numpy as np


class GlobalRNG:
    """
    a RNG that uses the default numpy primitives (e.g. np.random.randn but with the interface of a generator)
    this allows classes to instantiate a generator that is tied to the numpy global seed, so when the user calls
    np.random.seed(5) all instances with GlobalRNG as a generator will be influenced by the new seed
    (with np.random.default_rng() this is not the case)
    """

    def __getattr__(self, item):
        return getattr(np.random.random.__self__, item)

    @staticmethod
    def integers(low, high=None, size=None, dtype=None, endpoint=None):
        if endpoint:
            ints = np.random.random_integers(low, high, size)
            if dtype is not None:
                return ints.type(dtype)
            return ints
        return np.random.randint(low, high, size, dtype)