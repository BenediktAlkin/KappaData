import numpy as np
import torch


def getall_class_as_tensor(dataset):
    if hasattr(dataset, "getall_class"):
        classes = dataset.getall_class()
        if isinstance(classes, np.ndarray):
            return torch.from_numpy(classes)
        elif not torch.is_tensor(classes):
            return torch.tensor(classes)
    # slow sample-wise loading
    return torch.tensor([dataset.getitem_class(i) for i in range(len(dataset))])
