import numpy as np

from kappadata.datasets.kd_subset import KDSubset
from kappadata.utils.class_counts import get_class_counts_and_indices

class ClasswiseSubsetWrapper(KDSubset):
    def __init__(self, dataset, start_percent=None, end_percent=None):
        assert start_percent is not None or end_percent is not None
        assert start_percent is None or (isinstance(start_percent, (float, int)) and 0. <= start_percent <= 1.)
        assert end_percent is None or (isinstance(end_percent, (float, int)) and 0. <= end_percent <= 1.)
        start_percent = start_percent or 0.
        end_percent = end_percent or 1.
        assert start_percent <= end_percent

        counts, all_indices = get_class_counts_and_indices(dataset)
        sub_indices = []
        for i in range(dataset.n_classes):
            start_index = int(start_percent * counts[i])
            end_index = int(end_percent * counts[i])
            sub_indices += all_indices[i][start_index:end_index].tolist()

        super().__init__(dataset=dataset, indices=sub_indices)
