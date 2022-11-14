import torch


def get_random_bbox(h, w, lamb, rng):
    batch_size = len(lamb)
    bbox_hcenter = torch.randint(h, size=(batch_size,), generator=rng)
    bbox_wcenter = torch.randint(w, size=(batch_size,), generator=rng)

    area_half = 0.5 * (1.0 - lamb).sqrt()
    bbox_h_half = (area_half * h).floor()
    bbox_w_half = (area_half * w).floor()

    top = torch.clamp(bbox_hcenter - bbox_h_half, min=0).type(torch.long)
    bot = torch.clamp(bbox_hcenter + bbox_h_half, max=h).type(torch.long)
    left = torch.clamp(bbox_wcenter - bbox_w_half, min=0).type(torch.long)
    right = torch.clamp(bbox_wcenter + bbox_w_half, max=w).type(torch.long)
    bbox = torch.stack([top, left, bot, right], dim=1)

    lamb_adjusted = 1.0 - (bot - top) * (right - left) / (h * w)

    return bbox, lamb_adjusted


def cutmix_batch(x1, x2, bbox, inplace):
    if not inplace:
        x1 = x1.clone()
    for i in range(len(x1)):
        cutmix_sample_inplace(x1[i], x2[i], bbox[i])
    return x1


def cutmix_sample_inplace(x1, x2, bbox):
    top, left, bot, right = bbox
    x1[..., top:bot, left:right] = x2[..., top:bot, left:right]
    return x1
