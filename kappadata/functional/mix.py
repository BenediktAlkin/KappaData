import torch


def sample_lambda(alpha, size, rng):
    return torch.from_numpy(rng.beta(alpha, alpha, size=size)).type(torch.float32)


def sample_permutation(size, rng):
    return torch.from_numpy(rng.permutation(size)).type(torch.long)


def mix_y_idx2(y, idx2, lamb):
    if y is None:
        return y
    lamb_y = lamb.view(-1, 1)
    return y * lamb_y + y[idx2] * (1. - lamb_y)


def mix_y_y2(y, y2, lamb):
    return y * lamb + y2 * (1. - lamb)


def mix_y_inplace(y, lamb):
    if y is None:
        return y
    return y.roll(1, 0).mul_(1. - lamb).add_(y.mul(lamb))
