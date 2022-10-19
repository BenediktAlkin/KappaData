import einops
from .base.wrapper_base import WrapperBase
import numpy as np

class RepeatWrapper(WrapperBase):
    """ repeats the dataset <repetitions> times or until it reaches <size>"""

    def __init__(self, dataset, repetitions=None, size=None):
        assert (repetitions is not None) ^ (size is not None)
        assert len(dataset) > 0
        self.repetitions = repetitions
        self.size = size

        if repetitions is None:
            assert isinstance(size, int) and size > 0
            repetitions = np.ceil(size / len(dataset))

        # force at least 1 repetition
        repetitions = max(1, repetitions)
        # repeat indices <repetitions> times in round-robin fashion (indices are like [0, 1, 2, 0, 1, 2])
        indices = einops.repeat(torch.arange(len(dataset)), "i -> (r i)", r=repetitions)
        super().__init__(dataset=dataset, indices=indices)
