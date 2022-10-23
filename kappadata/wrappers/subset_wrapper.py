import numpy as np

from kappadata.datasets.kd_subset import KDSubset


class SubsetWrapper(KDSubset):
    def __init__(self, dataset, indices=None, start_index=None, end_index=None):
        if indices is None:
            # create indices from start/end index
            assert start_index is not None or end_index is not None
            assert start_index is None or isinstance(start_index, int)
            assert end_index is None or isinstance(end_index, int)
            end_index = end_index or len(dataset)
            end_index = min(end_index, len(dataset))
            start_index = start_index or 0
            indices = np.arange(start_index, end_index)
        else:
            assert start_index is None and end_index is None
        super().__init__(dataset=dataset, indices=indices)
