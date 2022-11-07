import torch
import einops
from kappadata.utils.param_checking import to_2tuple
from torchvision.transforms.functional import to_tensor


class UnpatchifyImage:
    def __init__(self, patch_size):
        patch_size = to_2tuple(patch_size)
        self.patch_h, self.patch_w = patch_size
        assert isinstance(self.patch_h, int) and self.patch_h > 0
        assert isinstance(self.patch_w, int) and self.patch_w > 0

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        return einops.rearrange(x, "(ph pw c) h w -> c (h ph) (w pw)", ph=self.patch_h, pw=self.patch_w)

