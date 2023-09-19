import numpy as np
import torch

from kappadata.datasets.kd_subset import KDSubset
from kappadata.utils.class_counts import get_class_counts
from kappadata.utils.getall_as_tensor import getall_as_tensor


class OversamplingWrapper(KDSubset):
    def __init__(self, dataset, mode="multiply"):
        self.mode = mode

        classes = getall_as_tensor(dataset)
        class_counts, _ = get_class_counts(classes, dataset.getdim_class())
        max_class_count = torch.max(class_counts).item()
        indices = torch.arange(len(dataset), dtype=torch.long)
        if self.mode == "multiply":
            # append miniority classes as long as they are not bigger than the majority class
            for i in range(len(class_counts)):
                # if class is not contained in dataset -> cant multiply sample
                if class_counts[i] == 0:
                    continue
                multiply_factor = int(np.floor(max_class_count / class_counts[i])) - 1
                if multiply_factor > 0:
                    # get indices of samples with class to oversample
                    all_indices = torch.arange(len(dataset), dtype=torch.long)
                    sample_idxs = all_indices[classes == i]
                    indices = torch.concat([indices, torch.tile(sample_idxs, dims=[multiply_factor])])
        elif self.mode == "exact":
            # add samples from minority classes until they have the same count as the majority class
            indices = []
            for i in range(len(class_counts)):
                remaining_indices = max_class_count
                indices_for_cur_class = (classes == i).nonzero().squeeze(1)
                while remaining_indices > 0:
                    perm = torch.arange(len(indices_for_cur_class))[:remaining_indices]
                    indices.append(indices_for_cur_class[perm])
                    remaining_indices -= len(perm)
            indices = torch.concat(indices)
        else:
            raise NotImplementedError(f"invalid oversampling mode '{self.mode}'")
        super().__init__(dataset=dataset, indices=indices)
