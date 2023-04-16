import matplotlib.pyplot as plt
import torch
from kappadata.visualization.visualize_masked_image import visualize_masked_image
from pathlib import Path
from torchvision.datasets.folder import default_loader
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--n_masks", default=100, type=int)
    return vars(parser.parse_args())


def main(root, mask_ratio, n_masks):
    # load image
    root = Path(root).expanduser()
    img = default_loader(root)

    # calculate mask length
    size = 300
    patch_size = 75
    seqlen_height = size // patch_size
    seqlen_width = size // patch_size
    seqlen = seqlen_height * seqlen_width

    temp_dir = Path("temp/MaskedImage")
    temp_dir.mkdir(exist_ok=True, parents=True)
    if mask_ratio == 0:
        n_masks = 1
    for i in range(n_masks):
        # generate mask
        seed = 0 + i
        mask_noise = torch.randn(seqlen, generator=torch.Generator().manual_seed(seed))
        ids_restore = torch.argsort(mask_noise)
        mask = torch.ones_like(mask_noise)
        mask[int(len(mask) * mask_ratio):] = 0
        mask = torch.gather(mask, dim=0, index=ids_restore)

        # visualize
        masked_img = visualize_masked_image(
            img,
            size=size,
            patch_size=patch_size,
            mask=mask,
            border=2,
            seed=seed,
            scale=(0.2, 1.0),
        )
        masked_img.save(temp_dir / f"{root.name}_{i}.png")


if __name__ == "__main__":
    main(**parse_args())