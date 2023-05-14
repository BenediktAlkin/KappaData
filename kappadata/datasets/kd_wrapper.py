from kappadata.errors import UseModeWrapperException
from .kd_dataset import KDDataset


class KDWrapper(KDDataset):
    def __init__(self, dataset: KDDataset, ctx_prefix: str = None):
        super().__init__()
        self.dataset = dataset
        self.ctx_prefix = ctx_prefix or type(self).__name__
        # children should overwrite _worker_init_fn
        assert type(self).worker_init_fn == KDWrapper.worker_init_fn

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    def __getitem__(self, idx):
        raise UseModeWrapperException

    def dispose(self):
        self.dataset.dispose()

    @property
    def root_dataset(self):
        # KDDataset implements root_dataset -> __getitem__ doesn't trigger
        return self.dataset.root_dataset

    @property
    def requires_propagate_ctx(self):
        return self.dataset.requires_propagate_ctx

    def has_wrapper(self, wrapper):
        if self == wrapper:
            return True
        return self.dataset.has_wrapper(wrapper)

    def has_wrapper_type(self, wrapper_type):
        if type(self) == wrapper_type:
            return True
        return self.dataset.has_wrapper_type(wrapper_type)

    @property
    def all_wrappers(self):
        return [self] + self.dataset.all_wrappers

    @property
    def all_wrapper_types(self):
        return [type(self)] + self.dataset.all_wrapper_types

    def get_wrappers_of_type(self, wrapper_type):
        wrappers = self.dataset.get_wrappers_of_type(wrapper_type)
        if type(self) == wrapper_type:
            return [self] + wrappers
        return wrappers

    def worker_init_fn(self, rank, **kwargs):
        self._worker_init_fn(rank, **kwargs)
        self.dataset.worker_init_fn(rank, **kwargs)

    def _worker_init_fn(self, rank, **kwargs):
        pass
