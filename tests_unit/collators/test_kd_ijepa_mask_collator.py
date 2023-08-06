import unittest

import numpy as np
import torch

from kappadata.collators.kd_ijepa_mask_collator import KDIjepaMaskCollator
from tests_external_sources.ijepa import MaskCollator
from tests_util.patch_rng import patch_rng


class TestKDIjepaMaskCollator(unittest.TestCase):
    @patch_rng(fn_names=["torch.randint"])
    def test_equal_to_original(self, seed):
        batch_size = 10
        input_size = 224
        patch_size = 16
        encoder_mask_scale = 0.85, 1.0
        predictor_mask_scale = 0.15, 0.2
        predictor_aspect_ratio = 0.75, 1.5
        num_enc_masks = 1
        num_pred_masks = 4
        min_keep = 10
        tries = 20
        kd_mask_collator = KDIjepaMaskCollator(
            input_size=input_size,
            patch_size=patch_size,
            encoder_mask_scale=encoder_mask_scale,
            predictor_mask_scale=predictor_mask_scale,
            predictor_aspect_ratio=predictor_aspect_ratio,
            num_enc_masks=num_enc_masks,
            num_pred_masks=num_pred_masks,
            min_keep=min_keep,
            tries=tries,
            dataset_mode="index x",
            return_ctx=True,
        )
        og_mask_collator = MaskCollator(
            input_size=input_size,
            patch_size=patch_size,
            enc_mask_scale=encoder_mask_scale,
            pred_mask_scale=predictor_mask_scale,
            aspect_ratio=predictor_aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            min_keep=min_keep,
            allow_overlap=False,
        )
        kd_mask_collator.rng = np.random.default_rng(seed=seed)
        kd_input = [((i, torch.full(size=(1, input_size, input_size), fill_value=i)), {}) for i in range(batch_size)]
        og_input = [torch.full(size=(1, input_size, input_size), fill_value=i) for i in range(batch_size)]
        for i in range(100):
            _, ctx = kd_mask_collator(kd_input)
            _, og_masks_encoder, og_masks_predictor = og_mask_collator(og_input)
            for j, og_mask_encoder in enumerate(og_masks_encoder):
                kd_mask_encoder = ctx["encoder_masks"][j * batch_size: (j + 1) * batch_size]
                self.assertTrue(torch.all(kd_mask_encoder == og_mask_encoder))
            for j, og_mask_predictor in enumerate(og_masks_predictor):
                kd_mask_predictor = ctx["predictor_masks"][j * batch_size: (j + 1) * batch_size]
                self.assertTrue(torch.all(kd_mask_predictor == og_mask_predictor))
