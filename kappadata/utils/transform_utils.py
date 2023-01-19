from functools import partial

from torchvision.transforms import Compose
from torchvision.transforms import Normalize

from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.norm.base.kd_norm_base import KDNormBase


def flatten_transform(transform):
    if transform is None:
        return []
    if isinstance(transform, (Compose, KDComposeTransform)):
        result = []
        for t in transform.transforms:
            result += flatten_transform(t)
        return result
    return [transform]


def get_norm_transform(transform):
    transforms = flatten_transform(transform)
    norm_transforms = [transform for transform in transforms if isinstance(transform, (Normalize, KDNormBase))]
    if len(norm_transforms) == 0:
        return None
    assert len(norm_transforms) == 1
    return norm_transforms[0]


def get_denorm_transform(transform, inplace=False):
    norm_transform = get_norm_transform(transform)
    if norm_transform is None:
        return None
    if isinstance(norm_transform, KDNormBase):
        return partial(norm_transform.denormalize, inplace=inplace)
    return Compose([
        Normalize(mean=(0., 0., 0.), std=tuple(1 / s for s in norm_transform.std)),
        Normalize(mean=tuple(-m for m in norm_transform.mean), std=(1., 1., 1.)),
    ])
