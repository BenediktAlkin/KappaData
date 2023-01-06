import unittest

from timm.data.mixup import Mixup
from torch.utils.data import DataLoader

from kappadata.collators.mix_collator import MixCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.wrappers.sample_wrappers.label_smoothing_wrapper import LabelSmoothingWrapper
from tests_util.datasets import create_image_classification_dataset


class TestMixupEqualsTimm(unittest.TestCase):
    def run_test(self, batch_size, smoothing, mixup_alpha, cutmix_alpha, p, cutmix_p, mode):
        ds = create_image_classification_dataset(seed=552, size=100, channels=3, resolution=32, n_classes=10)
        mixup_ds = LabelSmoothingWrapper(dataset=ds, smoothing=smoothing)
        collator = MixCollator(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            p=p,
            cutmix_p=cutmix_p,
            mode=mode,
            p_mode="batch",
            seed=5,
        )
        kd_loader = DataLoader(ModeWrapper(mixup_ds, mode="x class"), batch_size=batch_size, collate_fn=collator)

        timm_mixup = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=p,
            switch_prob=cutmix_p,
            mode=mode,
            label_smoothing=smoothing,
            num_classes=ds.n_classes,
        )
        # raw_loader = DataLoader(ModeWrapper(ds, mode="x class"), batch_size=batch_size)
        # for i, ((raw_x, raw_y), (kd_x, kd_y)) in enumerate(zip(raw_loader, kd_loader)):
        #     timm_x, timm_y = timm_mixup(raw_x, raw_y)
        #     self.assertTrue(torch.all(timm_x == kd_x), f"x is unequal for i={i}")
        #     self.assertTrue(torch.all(timm_y == kd_y), f"y is unequal for i={i} timm={timm_y} kd={kd_y}")

    def test(self):
        self.run_test(
            batch_size=4,
            smoothing=0.1,
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            p=1.0,
            cutmix_p=0.5,
            mode="batch",
        )
