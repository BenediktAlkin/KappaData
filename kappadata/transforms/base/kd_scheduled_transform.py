from kappaschedules import ScheduleBase, LinearIncreasing
from torch.utils.data import get_worker_info

from .kd_transform import KDTransform


class KDScheduledTransform(KDTransform):
    def __init__(self, transform: KDTransform, schedule: ScheduleBase = None):
        super().__init__()
        self.transform = transform
        self.rank = None
        self.num_workers = None
        self.batch_size = None
        self.n_batches = None
        self.sample_counter = 0
        # default to linear from 0 to 1
        self.schedule = schedule or LinearIncreasing()

    def worker_init_fn(
            self,
            rank,
            batch_size=None,
            dataset_length=None,
            drop_last=None,
            epochs=None,
            updates=None,
            samples=None,
            **__,
    ):
        self.rank = rank
        self.num_workers = get_worker_info().num_workers
        assert batch_size is not None
        self.batch_size = batch_size

        # calculate total_n_batches (number of batches independent of num_workers)
        if epochs is not None:
            assert updates is None and samples is None
            assert drop_last is not None
            assert dataset_length is not None
            assert isinstance(epochs, int) and epochs >= 0
            if drop_last:
                batches_per_epoch = dataset_length // batch_size
            else:
                batches_per_epoch = (dataset_length + batch_size - 1) // batch_size
            self.n_batches = epochs * batches_per_epoch
        elif updates is not None:
            assert samples is None
            assert isinstance(updates, int) and updates >= 0
            self.n_batches = updates
        elif samples is not None:
            assert isinstance(samples, int) and samples >= 0
            if samples % batch_size == 0:
                self.n_batches = samples // batch_size
            else:
                self.n_batches = samples // batch_size + 1
        else:
            raise NotImplementedError

    def __call__(self, x, ctx=None):
        # caulculate progress
        assert self.n_batches is not None, "call KDScheduledTransform.worker_init_fn before applying the transform"
        batch_idx = self.sample_counter // self.batch_size * self.num_workers + self.rank
        strength = self.schedule.get_value(batch_idx, self.n_batches)
        self.sample_counter += 1

        # scale
        self.transform.scale_strength(strength)

        return self.transform(x, ctx=ctx)
