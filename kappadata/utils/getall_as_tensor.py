import numpy as np
import torch

def getall(dataset, item="class"):
    getall_attr = f"getall_{item}"
    getitem_attr = f"getitem_{item}"

    if hasattr(dataset, getall_attr):
        # fast all-at-once loading
        return getattr(dataset, getall_attr)()
    if hasattr(dataset, getitem_attr):
        # slow sample-wise loading
        getitem = getattr(dataset, getitem_attr)
        return [getitem(i) for i in range(len(dataset))]
    raise NotImplementedError

def getall_as_list(dataset, item="class"):
    items = getall(dataset=dataset, item=item)
    if isinstance(items, list):
        return items
    if torch.is_tensor(items):
        return items.tolist()
    if isinstance(items, np.ndarray):
        return items.tolist()
    raise NotImplementedError

def getall_as_numpy(dataset, item="class"):
    items = getall(dataset=dataset, item=item)
    if isinstance(items, np.ndarray):
        return items
    elif not torch.is_tensor(items):
        items = torch.tensor(items)
    return items.numpy()

def getall_as_tensor(dataset, item="class"):
    items = getall(dataset=dataset, item=item)
    if isinstance(items, np.ndarray):
        return torch.from_numpy(items)
    elif not torch.is_tensor(items):
        return torch.tensor(items)
    return classes