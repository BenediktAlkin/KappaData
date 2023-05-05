from argparse import ArgumentParser
from pathlib import Path

import torch
from torchvision.datasets.folder import default_loader

from kappadata.visualization.visualize_mae_schematic import visualize_mae_schematic


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--seed", default=0, type=int)
    return vars(parser.parse_args())


def main(root, mask_ratio, seed):
    # load image
    root = Path(root).expanduser()
    img = default_loader(root)

    # calculate mask length
    size = 300
    patch_size = 75
    seqlen_height = size // patch_size
    seqlen_width = size // patch_size
    seqlen = seqlen_height * seqlen_width

    # generate mask
    mask_noise = torch.randn(seqlen, generator=torch.Generator().manual_seed(seed))
    ids_restore = torch.argsort(mask_noise)

    # visualize
    images = visualize_mae_schematic(
        img,
        ids_restore=ids_restore,
        size=size,
        patch_size=patch_size,
        border=2,
        mask_ratio=mask_ratio,
    )

    # save iamges
    temp_dir = Path("temp/MAE-schematic")
    temp_dir.mkdir(exist_ok=True, parents=True)
    for i, image in enumerate(images):
        image.save(temp_dir / f"{root.name}_{i}_{seed}.png")


if __name__ == "__main__":
    main(**parse_args())
