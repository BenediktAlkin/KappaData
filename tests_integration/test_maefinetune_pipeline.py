from torch.utils.data import DataLoader
import unittest
from unittest.mock import patch

import numpy as np
import torch
from timm.data import create_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.functional import to_pil_image, to_tensor

from kappadata.common.transforms.norm.kd_image_net_norm import KDImageNetNorm
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_rand_augment import KDRandAugment
from kappadata.transforms.kd_random_erasing import KDRandomErasing
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from tests_util.patch_rng import patch_rng
from timm.data import Mixup
from timm.data.mixup import cutmix_bbox_and_lam
from kappadata.wrappers.sample_wrappers.label_smoothing_wrapper import LabelSmoothingWrapper
from kappadata.collators.kd_mix_collator import KDMixCollator
from tests_util.datasets.classification_dataset_torch import ClassificationDatasetTorch
from tests_util.datasets.classification_dataset import ClassificationDataset
from kappadata.wrappers import ModeWrapper
from kappadata.wrappers.sample_wrappers import XTransformWrapper
from kappadata.utils.random import get_rng_from_global


class TestMaeFinetunePipeline(unittest.TestCase):
    def _run(self, images, classes, seed):
        batch_size = 4
        smoothing = 0.1
        mixup_alpha = 0.8
        cutmix_alpha = 1.0
        mixup_p = 0.5
        cutmix_p = 0.5
        mode = "batch"
        kd_dataset = ClassificationDataset(x=images, classes=classes)
        resolution = to_tensor(images[0]).shape[1]

        # create timm pipeline
        timm_transform = create_transform(
            input_size=resolution,
            is_training=True,
            color_jitter=None,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        timm_dataset = ClassificationDatasetTorch(x=images, classes=classes, transform=timm_transform)
        timm_mixup = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_p + cutmix_p,
            switch_prob=cutmix_p,
            mode=mode,
            label_smoothing=smoothing,
            num_classes=kd_dataset.getdim_class(),
        )

        # create KD pipeline
        kd_transform = KDComposeTransform([
            KDRandomResizedCrop(size=resolution, interpolation="bicubic"),
            KDRandomHorizontalFlip(),
            KDRandAugment(
                num_ops=2,
                magnitude=9,
                magnitude_std=0.5,
                fill_color=[124, 116, 104],
                interpolation="bicubic",
            ),
            KDImageNetNorm(),
            KDRandomErasing(p=0.25, mode="pixelwise", max_count=1),
        ])

        kd_dataset = XTransformWrapper(dataset=kd_dataset, transform=kd_transform)
        kd_dataset = LabelSmoothingWrapper(dataset=kd_dataset, smoothing=0.1)
        kd_dataset = ModeWrapper(dataset=kd_dataset, mode="x class", return_ctx=True)
        kd_collator = KDMixCollator(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_p=mixup_p,
            cutmix_p=cutmix_p,
            apply_mode=mode,
            lamb_mode=mode,
            shuffle_mode="flip",
            dataset_mode=kd_dataset.mode,
            return_ctx=True,
        )
        # set kd rng
        kd_rng = np.random.default_rng(seed=seed)
        kd_transform.set_rng(kd_rng)
        kd_collator.set_rng(kd_rng)
        # transforms/collators sample a seed from global numpy generator -> progress rng
        kd_rng.integers(np.iinfo(np.int32).max)  # KDRandomResizedCrop
        kd_rng.integers(np.iinfo(np.int32).max)  # KDRandomHorizontalFlip
        kd_rng.integers(np.iinfo(np.int32).max)  # KDRandAugment
        kd_rng.integers(np.iinfo(np.int32).max)  # KDRandomErasing
        kd_rng.integers(np.iinfo(np.int32).max)  # KDMixCollator

        # remember original methods that are patched
        timm_mixup.og_params_per_batch = timm_mixup._params_per_batch
        og_cutmix_bbox_and_lamb = cutmix_bbox_and_lam

        # patch _params_per_batch as KDMixCollator converts lambda to tensor which has precision errors
        def patch_params_per_batch(self_mixup):
            lam, use_cutmix = self_mixup.og_params_per_batch()
            return torch.tensor(lam), use_cutmix

        # patch mix_batch as KDMixCollator calculates lambda correction of CutMix as tensor -> precision errors
        def patch_cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
            (yl, yu, xl, xu), lam = og_cutmix_bbox_and_lamb(
                img_shape,
                lam,
                ratio_minmax=None,
                correct_lam=False,
                count=count,
            )
            if correct_lam or ratio_minmax is not None:
                bbox_area = (yu - yl) * (xu - xl)
                lam = 1. - torch.tensor(bbox_area) / float(img_shape[-2] * img_shape[-1])
            return (yl, yu, xl, xu), lam

        # create dataloaders
        timm_dataloader = DataLoader(timm_dataset, batch_size=batch_size)
        kd_dataloader = DataLoader(kd_dataset, batch_size=batch_size, collate_fn=kd_collator)

        for i, ((timm_x, timm_y), ((kd_x, kd_y), _)) in enumerate(zip(timm_dataloader, kd_dataloader)):
            with patch("timm.data.Mixup._params_per_batch", patch_params_per_batch):
                with patch("timm.data.mixup.cutmix_bbox_and_lam", patch_cutmix_bbox_and_lam):
                    timm_x, timm_y = timm_mixup(timm_x, timm_y)
            self.assertTrue(torch.all(timm_x == kd_x), str(i))
            self.assertTrue(torch.all(timm_y == kd_y), str(i))

    @staticmethod
    def get_patch_rng():
        return patch_rng(fn_names=[
            "random.uniform",
            "random.randint",
            "torch.rand",
            "random.random",
            "random.gauss",
            "numpy.random.choice",
            "torch.Tensor.normal_",
            "numpy.random.rand",
            "numpy.random.beta",
            "numpy.random.randint",
        ])

    def test_cifar10(self):
        images = [
            to_pil_image(x)
            for x in torch.rand(100, 3, 32, 32, generator=torch.Generator().manual_seed(513))
        ]
        classes = torch.randint(0, 10, size=(len(images),), generator=torch.Generator().manual_seed(905))
        with self.get_patch_rng() as prng:
            self._run(images=images, classes=classes, seed=prng.seed)

    def test_imagenet(self):
        images = [
            to_pil_image(x)
            for x in torch.rand(100, 3, 224, 224, generator=torch.Generator().manual_seed(513))
        ]
        classes = torch.randint(0, 1000, size=(len(images),), generator=torch.Generator().manual_seed(905))
        with self.get_patch_rng() as prng:
            self._run(images=images, classes=classes, seed=prng.seed)
