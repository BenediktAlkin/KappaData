import bisect
from functools import partial

from torch.utils.data import ConcatDataset

from kappadata.errors import UseModeWrapperException


class KDConcatDataset(ConcatDataset):
    def __getattr__(self, item):
        if item.startswith("getitem_"):
            # all methods starting with getitem_ are called with self.datasets[dataset_idx][sample_idx]
            return partial(self._call_getitem, item)
        if item == "datasets":
            return getattr(super(), item)
        if item.startswith("getall_"):
            # all methods starting with getall_ have to concatenate the result of dataset.getall_... for all datasets
            return partial(self._call_getall, item)
        # warning/exception here might make sense
        return getattr(self.datasets[0], item)

    def _call_getitem(self, item, idx, *args, **kwargs):
        dataset_idx, sample_idx = self._to_concat_idx(idx)
        func = getattr(self.datasets[dataset_idx], item)
        return func(sample_idx, *args, **kwargs)

    def _call_getall(self, item):
        result = []
        for dataset in self.datasets:
            dataset_result = getattr(dataset, item)()
            assert isinstance(dataset_result, list)
            result += dataset_result
        return result

    def _to_concat_idx(self, idx):
        """
        modification of __getitem__ from torch.utils.ConcatDataset that returns dataset_idx and sample_idx
        (instead of self.datasets[dataset_idx][sample_idx]
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()

    def dispose(self):
        for dataset in self.datasets:
            dataset.dispose()

    @property
    def root_dataset(self):
        if len(self.datasets) == 1:
            return self.datasets[0].root_dataset
        # warning/exception here might make sense
        return self.datasets[0].root_dataset

    def has_wrapper(self, wrapper):
        if len(self.datasets) == 1:
            return self.datasets[0].has_wrapper(wrapper)
        # warning/exception here might make sense
        return self.datasets[0].has_wrapper(wrapper)

    def has_wrapper_type(self, wrapper_type):
        if len(self.datasets) == 1:
            return self.datasets[0].has_wrapper_type(wrapper_type)
        # warning/exception here might make sense
        return self.datasets[0].has_wrapper_type(wrapper_type)

    @property
    def all_wrappers(self):
        if len(self.datasets) == 1:
            return self.datasets[0].all_wrappers
        # warning/exception here might make sense
        return self.datasets[0].all_wrappers

    @property
    def all_wrapper_types(self):
        if len(self.datasets) == 1:
            return self.datasets[0].all_wrapper_types
        # warning/exception here might make sense
        return self.datasets[0].all_wrapper_types

    def get_wrapper_of_type(self, wrapper_type):
        wrappers = self.get_wrappers_of_type(wrapper_type)
        if len(wrappers) == 0:
            return None
        assert len(wrappers) == 1
        return wrappers[0]

    def get_wrappers_of_type(self, wrapper_type):
        if len(self.datasets) == 1:
            return self.datasets[0].get_wrappers_of_type(wrapper_type)
        # warning/exception here might make sense
        return self.datasets[0].get_wrappers_of_type(wrapper_type)

    def worker_init_fn(self, rank, **kwargs):
        for dataset in self.datasets:
            dataset.worker_init_fn(rank, **kwargs)

    def __getitem__(self, idx):
        raise UseModeWrapperException
