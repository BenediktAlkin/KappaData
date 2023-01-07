from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path
from torchvision.transforms import Resize, ToPILImage
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.datasets import ImageFolder

import kappadata as kd
import torch
import random
import numpy as np
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--transform", type=str, required=True, choices=TRANSFORMS.keys())
    parser.add_argument("--collator", type=str, choices=COLLATORS.keys())
    parser.add_argument("--n_images", type=int, default=25)
    parser.add_argument("--n_augs_per_image", type=int, default=16)
    return vars(parser.parse_args())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def concat_images_square(images, padding):
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))

    w, h = images[0].size
    concated = Image.new(images[0].mode, (w * columns + padding * (columns - 1), h * rows + padding * (rows - 1)))
    for i in range(len(images)):
        col = (i % columns)
        row = i // columns
        concated.paste(images[i], (w * col + padding * (col - 1), h * row + padding * (row - 1)))
    return concated


TRANSFORMS = {
    "MAE-pretrain": kd.KDComposeTransform([
        kd.KDRandomResizedCrop(size=224, scale=(0.2, 1.0), interpolation="bicubic"),
        kd.KDRandomHorizontalFlip(),
    ]),
    "MAE-finetune": kd.KDComposeTransform([
        kd.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.KDRandomHorizontalFlip(),
        kd.KDRandAugment(
            num_ops=2,
            magnitude=9,
            magnitude_std=0.5,
            interpolation="bicubic",
            fill_color=(124, 116, 104),
        ),
        kd.KDImageNetNorm(),
        kd.KDRandomErasing(
            p=0.25,
            mode="pixelwise",
            max_count=1,
        ),
        kd.KDImageNetNorm(inverse=True),
        ToPILImage(),
    ]),
    "SSL-probe": kd.KDComposeTransform([
        kd.KDRandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation="bicubic"),
        kd.KDRandomHorizontalFlip(),
    ]),
}
COLLATORS = {
    "MAE-finetune": kd.KDMixCollator(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        mixup_p=0.5,
        cutmix_p=0.5,
        apply_mode="batch",
        lamb_mode="batch",
        shuffle_mode="random",
    )
}


def main(folder, transform, collator, n_images, n_augs_per_image):
    # setup
    set_seed(5)
    folder = Path(folder).expanduser()
    assert folder.exists()

    # initialize transforms
    noaug_transform = Resize(size=(224, 224))
    transform = TRANSFORMS[transform]

    # collect images
    ds = ImageFolder(root=folder)
    indices = torch.randperm(n_images).tolist()
    all_images = defaultdict(list)
    for idx in indices:
        x, _ = ds[idx]
        all_images[idx].append(noaug_transform(x))
        for i in range(n_augs_per_image - 1):
            all_images[idx].append(transform(x))

    # collate
    if collator is not None:
        collator = COLLATORS[collator]
        keys = list(all_images.keys())
        for i in range(1, n_augs_per_image):
            batch = [torch.stack([to_tensor(all_images[key][i]) for key in keys])]
            batch = collator.collate(batch, dataset_mode="x")
            for j, key in enumerate(keys):
                all_images[key][i] = to_pil_image(batch[0][j])

    # save images
    out_dir = Path("temp")
    out_dir.mkdir(exist_ok=True)
    for idx, images in all_images.items():
        img = concat_images_square(images=images, padding=2)
        img.save(out_dir / f"{idx}.png")


if __name__ == "__main__":
    main(**parse_args())
