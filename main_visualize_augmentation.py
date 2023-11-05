import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToPILImage
from torchvision.transforms.functional import to_tensor, to_pil_image

import kappadata as kd
from kappadata.utils.save_image import concat_images_square


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--transform", type=str, required=True, choices=list(TRANSFORMS.keys()) + ["all"])
    parser.add_argument("--collator", type=str, choices=list(COLLATORS.keys()) + ["all"])
    parser.add_argument("--n_images", type=int, default=25)
    parser.add_argument("--n_augs_per_image", type=int, default=16)
    return vars(parser.parse_args())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KDRandAugmentSingle(kd.transforms.KDRandAugment):
    def __init__(self, aug_idx, **kwargs):
        self.aug_idx = aug_idx
        super().__init__(num_ops=1, apply_op_p=1., **kwargs)

    def _get_ops(self):
        return [super()._get_ops()[self.aug_idx]]

    def _sample_transforms(self):
        return [self.ops[0]]


def get_randaug_transforms():
    return {
        f"RandAug{mag_name}-{key}": kd.transforms.KDComposeTransform([
            Resize(size=(224, 224)),
            KDRandAugmentSingle(
                aug_idx=i,
                magnitude=mag,
                magnitude_std=mag_std,
                interpolation="bicubic",
                fill_color=(124, 116, 104),
            ),
        ])
        for i, key in enumerate([
            "auto_contrast",
            "equalize",
            "invert",
            "rotate",
            "posterize",
            "solarize",
            "solarize_add",
            "color",
            "contrast",
            "brightness",
            "sharpness",
            "shear_x",
            "shear_y",
            "translate_horizontal",
            "translate_vertical",
        ])
        for mag_name, mag, mag_std in [
            ("(9,0.5)", 9, 0.5),
            ("(10,inf)", 10, float("inf")),
        ]
    }


TRANSFORMS = {
    "MAE-pretrain": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.2, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
    ]),
    "MAE-pretrain-normalizepixels": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.2, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
        kd.common.transforms.KDImageNetNorm(),
        kd.transforms.PatchifyImage(patch_size=16),
        kd.transforms.PatchwiseNorm(),
        kd.transforms.UnpatchifyImage(),
        kd.common.transforms.KDImageNetNorm(inverse=True),
        ToPILImage(),
    ]),
    "MAE-finetune": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
        kd.transforms.KDRandAugment(
            num_ops=2,
            magnitude=9,
            magnitude_std=0.5,
            interpolation="bicubic",
            fill_color=(124, 116, 104),
        ),
        kd.common.transforms.KDImageNetNorm(),
        kd.transforms.KDRandomErasing(
            p=0.25,
            mode="pixelwise",
            max_count=1,
        ),
        kd.common.transforms.KDImageNetNorm(inverse=True),
        ToPILImage(),
    ]),
    "MAE-finetune-randaugfirst": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandAugment(
            num_ops=2,
            magnitude=9,
            magnitude_std=0.5,
            interpolation="bicubic",
            fill_color=(124, 116, 104),
        ),
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
        kd.common.transforms.KDImageNetNorm(),
        kd.transforms.KDRandomErasing(
            p=0.25,
            mode="pixelwise",
            max_count=1,
        ),
        kd.common.transforms.KDImageNetNorm(inverse=True),
        ToPILImage(),
    ]),
    "SSL-probe": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
    ]),
    "BYOL-view0": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
        kd.transforms.KDRandomColorJitter(
            p=0.8,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        ),
        kd.transforms.KDGaussianBlurPIL(sigma=(0.1, 2.0)),
        kd.transforms.KDRandomGrayscale(p=0.2),
    ]),
    "BYOL-view0-torchvisionblur": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
        kd.transforms.KDRandomColorJitter(
            p=0.8,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        ),
        kd.transforms.KDGaussianBlurTV(kernel_size=23, sigma=(0.1, 2.0)),
        kd.transforms.KDRandomGrayscale(p=0.2),
    ]),
    "BYOL-view1": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
        kd.transforms.KDRandomColorJitter(
            p=0.8,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        ),
        kd.transforms.KDRandomGaussianBlurPIL(p=0.1, sigma=(0.1, 2.0)),
        kd.transforms.KDRandomGrayscale(p=0.2),
        kd.transforms.KDRandomSolarize(p=0.2, threshold=128),
    ]),
    "SIMCLR": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDRandomHorizontalFlip(),
        kd.transforms.KDRandomColorJitter(
            p=0.8,
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
        ),
        kd.transforms.KDRandomGaussianBlurPIL(p=0.5, sigma=(0.1, 2.0)),
        kd.transforms.KDRandomGrayscale(p=0.2),
    ]),
    "GaussianBlur-sigma=0.1": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDGaussianBlurPIL(sigma=0.1),
    ]),
    "GaussianBlur-sigma=1.0": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDGaussianBlurPIL(sigma=1.0),
    ]),
    "GaussianBlur-sigma=2.0": kd.transforms.KDComposeTransform([
        kd.transforms.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.transforms.KDGaussianBlurPIL(sigma=2.0),
    ]),
    **get_randaug_transforms(),
}
COLLATORS = {
    "MAE-finetune": kd.collators.KDMixCollator(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        mixup_p=0.5,
        cutmix_p=0.5,
        apply_mode="batch",
        lamb_mode="batch",
        shuffle_mode="random",
    )
}


def main_single(folder, transform, collator, n_images, n_augs_per_image):
    # setup
    set_seed(5)
    folder = Path(folder).expanduser()
    assert folder.exists()

    # initialize transforms
    noaug_transform = Resize(size=(224, 224))
    transform_name = transform
    transform = TRANSFORMS[transform]

    # collect images
    ds = ImageFolder(root=folder)
    indices = torch.randperm(len(ds))[:n_images].tolist()
    all_images = defaultdict(list)
    for idx in indices:
        x, _ = ds[idx]
        all_images[idx].append(noaug_transform(x))
        for i in range(n_augs_per_image - 1):
            all_images[idx].append(transform(x))

    # collate
    collator_name = None if collator is None else collator
    if collator is not None:
        collator = COLLATORS[collator]
        keys = list(all_images.keys())
        for i in range(1, n_augs_per_image):
            batch = [torch.stack([to_tensor(all_images[key][i]) for key in keys])]
            batch = collator.collate(batch, dataset_mode="x")
            for j, key in enumerate(keys):
                all_images[key][i] = to_pil_image(batch[0][j])

    # save images
    name = transform_name
    if collator_name is not None:
        name += f"--{collator_name}"
    out_dir = Path("temp") / name
    out_dir.mkdir(exist_ok=True, parents=True)
    for idx, images in all_images.items():
        img = concat_images_square(images=images, padding=2)
        img.save(out_dir / f"{idx}.png")


def main():
    args = parse_args()
    if args["transform"] == "all":
        transforms = list(TRANSFORMS.keys())
    else:
        transforms = [args["transform"]]
    if args["collator"] == "all":
        collators = [None] + list(COLLATORS.keys())
    else:
        collators = [args["collator"]]
    for transform in transforms:
        for collator in collators:
            args["transform"] = transform
            args["collator"] = collator
            main_single(**args)


if __name__ == "__main__":
    main()
