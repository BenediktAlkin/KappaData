import matplotlib.pyplot as plt
import torch
from kappadata.visualization.visualize_jigsaw import visualize_jigsaw
from pathlib import Path
from torchvision.datasets.folder import default_loader
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--repeat", default=100, type=int)
    return vars(parser.parse_args())


def main(root, repeat):
    # load image
    root = Path(root).expanduser()
    img = default_loader(root)

    size = 300
    patch_size = 75

    temp_dir = Path("temp/Jigsaw")
    temp_dir.mkdir(exist_ok=True, parents=True)
    for i in range(repeat):
        # visualize
        masked_img, perm = visualize_jigsaw(img, size=size, patch_size=patch_size, border=2, seed=i)
        print(f"{i}: {perm.tolist()}")
        masked_img.save(temp_dir / f"{root.name}_{i}.png")


if __name__ == "__main__":
    main(**parse_args())