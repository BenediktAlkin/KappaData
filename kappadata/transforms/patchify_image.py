import torch
import einops
from kappadata.utils.param_checking import to_2tuple
from torchvision.transforms.functional import to_tensor


class PatchifyImage:
    def __init__(self, patch_size):
        patch_size = to_2tuple(patch_size)
        self.patch_h, self.patch_w = patch_size
        assert isinstance(self.patch_h, int) and self.patch_h > 0
        assert isinstance(self.patch_w, int) and self.patch_w > 0

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        c, src_h, src_w = x.shape
        assert src_h % self.patch_h == 0 and src_w % self.patch_w == 0
        # how many patches are along height/width dimension
        lh = src_h // self.patch_h
        lw = src_w // self.patch_w
        # reshape to patches
        x = einops.rearrange(x, "c (lh ph) (lw pw) -> (ph pw c) lh lw", lh=lh, ph=self.patch_h, lw=lw, pw=self.patch_w)
        return x

