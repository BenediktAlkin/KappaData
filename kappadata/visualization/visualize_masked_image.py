import torch
import einops
from torchvision.transforms.functional import to_tensor, resize, to_pil_image, InterpolationMode
from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.unpatchify_image import UnpatchifyImage
from kappadata.utils.param_checking import to_2tuple

def visualize_masked_image(img, size=300, patch_size=75, mask=None, border=2, fill="gray"):
    if not torch.is_tensor(img):
        img = to_tensor(img)
    assert img.ndim == 3
    assert mask.ndim == 1

    # preprocess
    img = resize(img, size=to_2tuple(size), interpolation=InterpolationMode.BILINEAR)

    # patchify
    ctx = {}
    patches = PatchifyImage(patch_size=to_2tuple(patch_size))(img, ctx=ctx)
    assert patches.size(1) == mask.size(0)

    # create background
    if fill == "black":
        fill_value = 0.
    elif fill == "gray":
        fill_value = 0.5
    elif fill == "white":
        fill_value = 1.
    else:
        raise NotImplementedError
    background = torch.full_like(patches, fill_value=fill_value)

    # mask out
    mask = einops.rearrange(mask, "l -> 1 l 1 1")
    patches = patches * (1 - mask)
    background = background * mask
    patches = patches + background

    # make border
    if border > 0:
        patches[:, :, :border] = 1.
        patches[:, :, -border:] = 1.
        patches[:, :, :, :border] = 1.
        patches[:, :, :, -border:] = 1.

    # unpatchify
    img = UnpatchifyImage()(patches, ctx=ctx)

    # to image
    return to_pil_image(img)
