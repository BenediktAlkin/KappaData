def mixup_roll(x, lamb):
    return x.roll(1, 0).mul_(1.0 - lamb).add_(x.mul(lamb))


def mixup_idx2(x, idx2, lamb):
    return x[idx2].mul_(1.0 - lamb).add_(x.mul(lamb))


def mixup_inplace(x1, x2, lamb):
    return x1.mul_(1.0 - lamb).add_(x2.mul(lamb))
