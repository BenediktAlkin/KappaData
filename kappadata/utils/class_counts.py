import numpy as np
import torch


def get_class_counts(classes, n_classes):
    if n_classes == 1:
        n_classes = 2

    # torch has much better asymptotic complexity (noticable from 1e9)
    if isinstance(classes, np.ndarray):
        classes = torch.from_numpy(classes).long()
    else:
        classes = torch.tensor(classes, dtype=torch.long)
    # it is much faster on GPU, but also requires a lot of memory for large numbers
    counts = torch.zeros(n_classes, dtype=torch.long)
    unique_classes, unique_counts = classes.unique(return_counts=True)
    counts[unique_classes] = unique_counts
    return counts


def get_class_counts_from_dataset(dataset):
    classes = [dataset.getitem_class(i) for i in range(len(dataset))]
    return get_class_counts(classes=classes, n_classes=dataset.n_classes)


def get_class_counts_and_indices(dataset):
    # TODO inefficient implementation (e.g. https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array)
    classes = np.array([dataset.getitem_class(i) for i in range(len(dataset))])
    counts = get_class_counts(classes=classes, n_classes=dataset.n_classes)
    indices = []
    for i in range(dataset.n_classes):
        indices.append((classes == i).nonzero()[0])
    return counts, indices
