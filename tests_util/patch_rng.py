import torch
from unittest.mock import patch
import numpy as np
from contextlib import ContextDecorator


# TODO refactor tests to use this
class patch_rng(ContextDecorator):
    def __init__(self, fn_names, seed=5):
        super().__init__()
        self.ctx_managers = []
        rng = np.random.default_rng(seed=seed)
        for fn_name in fn_names:
            if fn_name == "numpy.random.beta":
                ctx_manager = patch("numpy.random.beta", lambda a, b: rng.beta(a, b))
            elif fn_name == "numpy.random.choice":
                ctx_manager = patch("numpy.random.choice", rng.choice)
            elif fn_name == "numpy.random.rand":
                ctx_manager = patch("numpy.random.rand", rng.random)
            elif fn_name == "numpy.random.randint":
                ctx_manager = patch("numpy.random.randint", lambda a, b, size: rng.integers(a, b, size=size))
            elif fn_name == "random.gauss":
                ctx_manager = patch("random.gauss", lambda mu, sigma: rng.normal(mu, sigma))
            elif fn_name == "random.randint":
                ctx_manager = patch(
                    "random.randint",
                    lambda low, high: int(rng.integers(low, high + 1)) if low != high else low,
                )
            elif fn_name == "random.random":
                ctx_manager = patch("random.random", rng.random)
            elif fn_name == "random.uniform":
                ctx_manager = patch("random.uniform", lambda low, high: rng.uniform(low, high))
            elif fn_name == "torch.rand":
                ctx_manager = patch("torch.rand", lambda _: torch.tensor(rng.random(), dtype=torch.float64))
            elif fn_name == "torch.randint":
                ctx_manager = patch(
                    "torch.randint",
                    lambda low, high, size: torch.tensor([int(rng.integers(low, high))]),
                )
            elif fn_name == "torch.randperm":
                ctx_manager = patch("torch.randperm", lambda n: torch.tensor(rng.permutation(n)))
            elif fn_name == "torch.Tensor.normal_":
                ctx_manager = patch(
                    "torch.Tensor.normal_",
                    lambda tensor: torch.from_numpy(rng.standard_normal(size=tensor.shape, dtype=np.float32)),
                )
            elif fn_name == "torch.Tensor.uniform_":
                ctx_manager = patch(
                    "torch.Tensor.uniform_",
                    lambda _, low, high: torch.tensor([rng.uniform(low, high)], dtype=torch.float64),
                )
            else:
                raise NotImplementedError(fn_name)
            self.ctx_managers.append(ctx_manager)

    def __enter__(self):
        for ctx_manager in self.ctx_managers:
            ctx_manager.__enter__()
        return self

    def __exit__(self, *args):
        for ctx_manager in self.ctx_managers:
            ctx_manager.__exit__(*args)
