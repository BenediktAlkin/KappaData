import numpy as np
from torch.nn.functional import one_hot


def get_random_bbox(h, w, lamb, rng):
    # TODO batch version of this
    #  (should use torch for it but torch.randint might have different impl than rng.intergers)
    cut_ratio = np.sqrt(1. - lamb)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    h_center = rng.integers(h)
    w_center = rng.integers(w)

    left = np.clip(w_center - cut_w // 2, 0, w)
    right = np.clip(w_center + cut_w // 2, 0, w)
    top = np.clip(h_center - cut_h // 2, 0, h)
    bot = np.clip(h_center + cut_h // 2, 0, h)

    return top, left, bot, right


def get_area_of_bbox(bbox, h, w):
    top, left, bot, right = bbox
    return 1 - (right - left) * (bot - top) / (h * w)


def to_onehot_vector(y, n_classes):
    if y.ndim == 0:
        y = one_hot(y, num_classes=n_classes)
    return y


def to_onehot_matrix(y, n_classes):
    if y.ndim == 1:
        y = one_hot(y, num_classes=n_classes)
    return y


def cutmix_batch(x1, x2, bbox):
    for i in range(len(x1)):
        cutmix_single(x1=x1[i], x2=x2[i], bbox=bbox[i])
    return x1


def cutmix_single(x1, x2, bbox):
    top, left, bot, right = bbox
    x1[:, top:bot, left:right] = x2[:, top:bot, left:right]
    return x1
