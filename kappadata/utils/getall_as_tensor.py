import numpy as np
import torch


def getall_as_tensor(dataset, item="class"):
    getall_attr = f"getall_{item}"
    getitem_attr = f"getitem_{item}"

    # load classes
    if hasattr(dataset, getall_attr):
        # fast all-at-once loading
        classes = getattr(dataset, getall_attr)()
    elif hasattr(dataset, getitem_attr):
        # slow sample-wise loading
        getitem = getattr(dataset, getitem_attr)
        classes = [getitem(i) for i in range(len(dataset))]
    else:
        raise NotImplementedError

    # convert to tensor
    if isinstance(classes, np.ndarray):
        return torch.from_numpy(classes)
    elif not torch.is_tensor(classes):
        return torch.tensor(classes)
    raise NotImplementedError