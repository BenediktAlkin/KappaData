import numpy as np


class GlobalRng:
    def __getattr__(self, item):
        return getattr(np.random, item)

    @staticmethod
    def integers(low, high=None, size=None):
        return np.random.randint(low=low, high=high, size=size)
