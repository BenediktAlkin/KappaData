import numpy as np


def get_class_counts(classes, n_classes):
    if n_classes == 1:
        n_classes = 2
    assert len(classes) > 0 and n_classes > 1
    unique_classes, unique_counts = np.unique(classes, return_counts=True)
    # classes might have 0 samples
    counts = np.zeros(n_classes, dtype=int)
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