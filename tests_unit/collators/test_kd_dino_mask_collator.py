import torch
import unittest

from kappadata.collators.kd_dino_mask_collator import KDDinoMaskCollator
import random
import math
import numpy as np
from tests_external_sources.dinov2 import MaskingGenerator, collate_data_and_cast
from tests_util.patch_rng import patch_rng

class TestKDDinoMaskCollator(unittest.TestCase):
    @patch_rng(fn_names=["random.uniform", "random.shuffle", "random.randint"])
    def test_equal_to_original(self, seed):
        batch_size = 10
        global_size = 224
        local_size = 96
        patch_size = 16
        num_global_views = 2
        num_local_views = 8
        mask_collator = KDDinoMaskCollator(
            mask_ratio=(0.1, 0.5),
            mask_prob=0.5,
            mask_size=global_size // patch_size,
            dataset_mode="x",
            return_ctx=False,
        )
        mask_collator.rng = np.random.default_rng(seed=seed)
        for i in range(100):
            kd_result = mask_collator(list(range(batch_size * 2)))

            collator_input = [
                [
                    dict(
                        global_crops=[
                            torch.full(size=(1, global_size, global_size), fill_value=i)
                            for i in range(num_global_views)
                        ],
                        global_crops_teacher=[torch.zeros(1, global_size, global_size) for _ in range(num_global_views)],
                        local_crops=[torch.zeros(1, local_size, local_size) for _ in range(num_local_views)],
                        offsets=(),
                    )
                ]
                for _ in range(batch_size)
            ]
            result = collate_data_and_cast(
                samples_list=collator_input,
                mask_ratio_tuple=(0.1, 0.5),
                mask_probability=0.5,
                n_tokens=(global_size // patch_size) ** 2,
                mask_generator=MaskingGenerator(
                    input_size=global_size // patch_size,
                    max_num_patches=0.5 * global_size // patch_size * global_size // patch_size,
                ),
            )
            self.assertTrue(torch.all(result["collated_masks"] == kd_result.flatten(start_dim=1)), str(i))

