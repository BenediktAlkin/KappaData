from torch.nn.functional import one_hot


def to_onehot_vector(y, n_classes):
    if y.ndim == 0:
        y = one_hot(y, num_classes=n_classes)
    return y


def to_onehot_matrix(y, n_classes):
    if y.ndim == 1:
        y = one_hot(y, num_classes=n_classes)
    return y
