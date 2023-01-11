import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter

from .base.kd_stochastic_transform import KDStochasticTransform


class KDColorJitter(KDStochasticTransform):
    def __init__(self, brightness, contrast, saturation, hue, **kwargs):
        super().__init__(**kwargs)
        # ColorJitter preprocesses the parameters -> just use original implementation to store parameters
        self.tv_colorjitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, x, ctx=None):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params()
        # if ctx is not None:
        #     ctx["color_jitter"] = dict(
        #         fn_idx=fn_idx.tolist(),
        #         brightness_factor=brightness_factor,
        #         contrast_factor=contrast_factor,
        #         saturation_factor=saturation_factor,
        #         hue_factor=hue_factor,
        #     )
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                x = F.adjust_brightness(x, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                x = F.adjust_contrast(x, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                x = F.adjust_saturation(x, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                x = F.adjust_hue(x, hue_factor)
        return x

    def get_params(self):
        brightness = self.tv_colorjitter.brightness
        contrast = self.tv_colorjitter.contrast
        saturation = self.tv_colorjitter.saturation
        hue = self.tv_colorjitter.hue

        fn_idx = self.rng.permutation(4)

        b = None if brightness is None else self.rng.uniform(brightness[0], brightness[1])
        c = None if contrast is None else self.rng.uniform(contrast[0], contrast[1])
        s = None if saturation is None else self.rng.uniform(saturation[0], saturation[1])
        h = None if hue is None else self.rng.uniform(hue[0], hue[1])

        return fn_idx, b, c, s, h
