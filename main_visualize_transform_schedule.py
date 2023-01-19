from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from kappaschedules import LinearIncreasing, CosineIncreasing
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

from kappadata.common.datasets import KDImageFolder
from kappadata.transforms import KDSolarize, KDScheduledTransform, KDComposeTransform
from kappadata.utils.save_image import concat_images_square
from kappadata.wrappers import ModeWrapper
from kappadata.wrappers.dataset_wrappers import SubsetWrapper


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    return parser.parse_args()


def main(root):
    root = Path(root).expanduser()
    for name, schedule in [
        ("linear", LinearIncreasing()),
        ("cosine", CosineIncreasing()),
    ]:
        transform = KDScheduledTransform(
            transform=KDComposeTransform([
                Resize(size=(224, 224)),
                KDSolarize(threshold=0),
                ToTensor(),
            ]),
            schedule=schedule,
        )
        dataset = ModeWrapper(
            dataset=SubsetWrapper(dataset=KDImageFolder(root, transform=transform), indices=[0] * 16),
            mode="x",
            return_ctx=True,
        )
        worker_init_fn = partial(
            transform.worker_init_fn,
            batch_size=1,
            dataset_length=len(dataset),
            drop_last=False,
            epochs=1,
        )
        loader = DataLoader(dataset=dataset, batch_size=1, worker_init_fn=worker_init_fn, num_workers=1)
        images = []
        for i, batch in enumerate(loader):
            x, ctx = batch
            print(f"{i}: {ctx['KDSolarize.threshold'].item()}")
            images.append(to_pil_image(x[0]))

        # save images
        out_dir = Path("temp") / "schedule"
        out_dir.mkdir(exist_ok=True, parents=True)
        img = concat_images_square(images=images, padding=2)
        img.save(out_dir / f"{name}.png")


if __name__ == "__main__":
    main(**vars(parse_args()))
