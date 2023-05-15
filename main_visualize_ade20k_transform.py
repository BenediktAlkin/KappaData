from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets.folder import default_loader

from kappadata.visualization.visualize_jigsaw import visualize_jigsaw
import torch
import numpy as np
from PIL import Image
import os
from kappadata.utils.hash_utils import hash_tensor_entries

from kappadata.transforms import (
    KDSemsegRandomHorizontalFlip,
    KDSemsegPad,
    KDSemsegRandomResize,
    KDSemsegRandomCrop,
    KDColorJitter,
    KDImageRangeNorm,
    KDStochasticTransform,
)
from kappadata.wrappers import SemsegTransformWrapper, ModeWrapper
from kappadata.datasets import KDDataset
from torchvision.transforms.functional import to_pil_image

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--repeat", default=100, type=int)
    return vars(parser.parse_args())

class SemsegDataset(KDDataset):
    def __init__(self, x, semseg):
        super().__init__()
        self.x = x
        self.semseg = semseg

    # noinspection PyUnusedLocal
    def getitem_x(self, idx, ctx=None):
        return self.x[idx]

    # noinspection PyUnusedLocal
    def getitem_semseg(self, idx, ctx=None):
        return self.semseg[idx]

    def __len__(self):
        return len(self.x)

def main(root, repeat):
    root = Path(root).expanduser()
    root_imgs = root / "images"
    root_masks = root / "annotations"
    # load first image
    fname = os.listdir(root_imgs / "validation")[0][:-len(".jpg")]
    img = default_loader(root_imgs / "validation" / f"{fname}.jpg")
    mask = torch.from_numpy(np.array(Image.open(root_masks / "validation" / f"{fname}.png")).astype('int32')) - 1

    ds = SemsegTransformWrapper(
        dataset=SemsegDataset([img], [mask]),
        transforms=[
            KDSemsegRandomResize(base_size=(2048, 512), ratio=(0.5, 2.0), interpolation="bicubic"),
            KDSemsegRandomCrop(size=512, max_category_ratio=0.75),
            KDSemsegRandomHorizontalFlip(),
            KDColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            KDSemsegPad(size=512),
        ],
    )
    ds = ModeWrapper(dataset=ds, mode="x semseg", return_ctx=False)

    temp_dir = Path("temp/ADE20K")
    temp_dir.mkdir(exist_ok=True, parents=True)
    for i in range(repeat):
        print(i)
        for t in ds.transforms:
            if isinstance(t, KDStochasticTransform):
                t.set_rng(np.random.default_rng(i + 39824))

        if i == 0:
            # load unaugmented image
            x = ds.dataset.getitem_x(0)
            semseg = ds.dataset.getitem_semseg(0)
        else:
            x, semseg = ds[0]
        if torch.is_tensor(x):
            x = to_pil_image(x)
        x.save(temp_dir / f"{fname}_{i}.jpg")
        semseg += 1
        semseg_r = hash_tensor_entries(semseg, shuffle_primes_seed=0) % 255
        semseg_g = hash_tensor_entries(semseg, shuffle_primes_seed=1) % 255
        semseg_b = hash_tensor_entries(semseg, shuffle_primes_seed=2) % 255
        semseg = torch.stack([semseg_r, semseg_g, semseg_b])
        to_pil_image(semseg / 255).save(temp_dir / f"{fname}_{i}.png")


if __name__ == "__main__":
    main(**parse_args())
