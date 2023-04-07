import numpy as np
import torch

from kappadata.datasets.kd_subset import KDSubset
from kappadata.utils.class_counts import get_class_counts


class OversamplingWrapper(KDSubset):
    def __init__(self, dataset, strategy="multiply"):
        self.strategy = strategy

        if hasattr(dataset, "getall_class"):
            classes = dataset.getall_class()
            if isinstance(classes, np.ndarray):
                classes = torch.from_numpy(classes)
            elif not torch.is_tensor(classes):
                classes = torch.tensor(classes)
        else:
            classes = torch.tensor([dataset.getitem_class(i) for i in range(len(dataset))])
        class_counts = get_class_counts(classes, dataset.n_classes)
        max_class_count = torch.max(class_counts)
        indices = torch.arange(len(dataset), dtype=torch.long)
        if self.strategy == "multiply":
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
        else:
            raise NotImplementedError(f"invalid oversampling strategy {self.strategy}")
        super().__init__(dataset=dataset, indices=indices.tolist())
