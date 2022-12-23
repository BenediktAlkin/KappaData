import argparse
import logging
import shutil
import sys
from functools import partial
from pathlib import Path

# noinspection PyPackageRequirements
import kappaprofiler as kp
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, CenterCrop, Compose

import kappadata as kd
from kappadata.loading.image_folder import raw_image_loader, raw_image_folder_sample_to_pil_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    return vars(parser.parse_args())


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt=f"%(asctime)s %(levelname).1s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    ))
    logger.handlers = [handler]


class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


def main(image_folder_path, epochs, batch_size, num_workers):
    setup_logging()
    image_folder_path = Path(image_folder_path).expanduser()
    assert image_folder_path.exists()
    transform = Compose([CenterCrop(size=224), ToTensor()])

    caching_datasets = {
        "ImageFolder": None,
        "SharedDictDataset": kd.SharedDictDataset,
    }
    if shutil.which("redis-server") is not None:
        caching_datasets["RedisDataset"] = partial(kd.RedisDataset, port=55558)

    for name, caching_dataset in caching_datasets.items():
        if caching_dataset is not None:
            image_folder = ImageFolder(image_folder_path, loader=raw_image_loader)
            cached_dataset = caching_dataset(dataset=image_folder, transform=raw_image_folder_sample_to_pil_sample)
            dataset = DatasetWrapper(cached_dataset, transform=transform)
        else:
            dataset = ImageFolder(image_folder_path, transform=transform)

        logging.info(f"benchmarking {name}")
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=num_workers > 0)
        # spawn workers
        _ = iter(loader)
        n_batches = len(loader)
        with kp.Stopwatch() as sw:
            for i in range(epochs):
                with kp.Stopwatch() as epoch_sw:
                    for j, (_, _) in enumerate(loader):
                        print(f"{j + 1}/{n_batches}", end="\r")
                logging.info(f"epoch {i} took: {epoch_sw.elapsed_seconds}")
        logging.info(f"{epochs} epochs took: {sw.elapsed_seconds}")
        if caching_dataset is not None:
            dataset.dispose()


if __name__ == "__main__":
    main(**parse_args())
