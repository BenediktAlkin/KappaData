import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch.cuda
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from kappadata.visualization.visualize_dataset_imgsize import visualize_dataset_imgsize


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    return vars(parser.parse_args())


class WalkdirDataset(Dataset):
    def __init__(self, root, extensions=None):
        super().__init__()
        root = Path(root).expanduser()
        assert root.exists()
        extensions = [ext.replace(".", "") for ext in extensions or []]
        self.paths = []
        for froot, _, fnames in os.walk(root):
            self.paths += [Path(froot) / fname for fname in fnames if fname.split(".")[-1] in extensions]

    def __getitem__(self, idx):
        return default_loader(self.paths[idx])

    def __len__(self):
        return len(self.paths)


def main(root):
    dataset = WalkdirDataset(Path(root).expanduser(), extensions=["jpg"])
    bucket_size = 30
    height_data, width_data, area_data = visualize_dataset_imgsize(
        dataset,
        cutoff=0.05,
        bucket_size=bucket_size,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    for name, data in [("height", height_data), ("width", width_data)]:
        plt.clf()
        plt.bar(range(data.start, data.end + 1, bucket_size), data.values, width=bucket_size)
        plt.xlabel(f"{name} avg={data.avg:.1f}")
        plt.ylabel("histogram [%]")
        plt.title(f"{root} contains {len(dataset)} images")
        plt.show()


if __name__ == "__main__":
    main(**parse_args())
