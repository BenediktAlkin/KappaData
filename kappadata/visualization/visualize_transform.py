import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image, InterpolationMode

from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform
from kappadata.utils.param_checking import to_2tuple


def visualize_transform(img, transform, size=300, seed=0):
    assert isinstance(img, Image.Image) or (torch.is_tensor(img) and img.ndim == 3)

    # preprocess
    img = resize(img, size=to_2tuple(size), interpolation=InterpolationMode.BILINEAR)

    # apply transform
    rng = np.random.default_rng(seed=seed)
    if isinstance(transform, (KDStochasticTransform, KDComposeTransform)):
        transform.set_rng(rng)
    img = transform(img)

    # to image
    if torch.is_tensor(img):
        img = to_pil_image(img)
    return img
