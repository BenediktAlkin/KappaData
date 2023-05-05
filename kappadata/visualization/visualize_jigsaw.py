import torch
from torchvision.transforms.functional import to_tensor, resize, to_pil_image, InterpolationMode

from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.unpatchify_image import UnpatchifyImage
from kappadata.utils.param_checking import to_2tuple


def visualize_jigsaw(img, size=300, patch_size=75, border=2, seed=0):
    if not torch.is_tensor(img):
        img = to_tensor(img)
    assert img.ndim == 3

    # preprocess
    img = resize(img, size=to_2tuple(size), interpolation=InterpolationMode.BILINEAR)

    # patchify
    ctx = {}
    patches = PatchifyImage(patch_size=to_2tuple(patch_size))(img, ctx=ctx)

    # shuffle
    perm = torch.randperm(patches.size(1), generator=torch.Generator().manual_seed(seed))
    patches = patches[:, perm]

    # make border
    if border > 0:
        patches[:, :, :border] = 1.
        patches[:, :, -border:] = 1.
        patches[:, :, :, :border] = 1.
        patches[:, :, :, -border:] = 1.

    # unpatchify
    img = UnpatchifyImage()(patches, ctx=ctx)

    # to image
    return to_pil_image(img), perm
