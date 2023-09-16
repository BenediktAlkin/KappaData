import unittest

import numpy as np

from kappadata.utils.global_rng import GlobalRng
from tests_util.patch_rng import patch_rng


class TestGlobalRng(unittest.TestCase):
    def test_permuted(self):
        rng = GlobalRng()
        x = np.arange(10)
        seed = 842
        with patch_rng(fn_names=["numpy.random.permutation"], seed=seed):
            grng = rng.permuted(x)
        srng = np.random.default_rng(seed=seed).permuted(x)
        self.assertEqual(grng.tolist(), srng.tolist())
