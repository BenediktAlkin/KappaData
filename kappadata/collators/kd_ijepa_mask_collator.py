from .base import KDSingleCollator
import einops
from torchvision.transforms.functional import to_pil_image
import math

from multiprocessing import Value

import torch

from kappadata.utils.param_checking import to_2tuple
from kappadata.wrappers import ModeWrapper


class KDIjepaMaskCollator(KDSingleCollator):
    """ KappaData adaption of https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py """

    def __init__(
            self,
            input_size=(224, 224),
            patch_size=16,
            encoder_mask_scale=(0.85, 1.0),
            predictor_mask_scale=(0.15, 0.2),
            predictor_aspect_ratio=(0.75, 1.5),
            num_enc_masks=1,
            num_pred_masks=4,
            min_keep=10,
            tries=20,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = to_2tuple(input_size)
        self.patch_size = to_2tuple(patch_size)
        self.seqlen_h = self.input_size[0] // self.patch_size[0]
        self.seqlen_w = self.input_size[1] // self.patch_size[1]
        self.encoder_mask_scale = to_2tuple(encoder_mask_scale)
        self.predictor_mask_scale = to_2tuple(predictor_mask_scale)
        self.predictor_aspect_ratio = to_2tuple(predictor_aspect_ratio)
        self.num_enc_masks = num_enc_masks
        self.num_pred_masks = num_pred_masks
        self.min_keep = min_keep
        self.tries = tries
        self._itr_counter = Value('i', -1)

    @property
    def default_collate_mode(self):
        return "before"

    def collate(self, batch, dataset_mode, ctx=None):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """

        if ctx is None:
            return batch
        x = ModeWrapper.get_item(mode=dataset_mode, item="x", batch=batch)
        batch_size = len(x)

        # sample mask sizes for predictor and encoder (enforce same size for all GPUs via seed)
        seed = self.step()
        generator = torch.Generator().manual_seed(seed)
        predictor_size = self._sample_block_size(
            generator=generator,
            scale=self.predictor_mask_scale,
            aspect_ratio_range=self.predictor_aspect_ratio,
        )
        encoder_size = self._sample_block_size(
            generator=generator,
            scale=self.encoder_mask_scale,
            aspect_ratio_range=(1., 1.),
        )

        # generate masks
        masks_predictor, masks_encoder = [], []
        min_keep_pred = min_keep_enc = self.seqlen_h * self.seqlen_w
        for _ in range(batch_size):
            # predictor masks
            masks_pred, masks_pred_complement = [], []
            for _ in range(self.num_pred_masks):
                mask, mask_complement = self._sample_block_mask(predictor_size)
                masks_pred.append(mask)
                masks_pred_complement.append(mask_complement)
                min_keep_pred = min(min_keep_pred, len(mask))
            masks_predictor.append(masks_pred)
            # encoder masks
            masks_enc = []
            for _ in range(self.num_enc_masks):
                mask = self._sample_block_mask_constrained(encoder_size, acceptable_regions=masks_pred_complement)
                masks_enc.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            masks_encoder.append(masks_enc)

        # collate masks
        masks_predictor = [[mask[:min_keep_pred] for mask in masks] for masks in masks_predictor]
        masks_predictor = torch.utils.data.default_collate(masks_predictor)
        masks_encoder = [[mask[:min_keep_enc] for mask in masks] for masks in masks_encoder]
        masks_encoder = torch.utils.data.default_collate(masks_encoder)

        ctx["masks_encoder"] = masks_encoder
        ctx["masks_predictor"] = masks_predictor
        return batch

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_range):
        rand = torch.rand(1, generator=generator).item()
        # sample block scale
        min_scale, max_scale = scale
        mask_scale = min_scale + rand * (max_scale - min_scale)
        max_keep = int(self.seqlen_h * self.seqlen_w * mask_scale)
        # sample block aspect-ratio (not sure why they dont sample in log-space like RandomResizedCrop does it)
        min_ar, max_ar = aspect_ratio_range
        aspect_ratio = min_ar + rand * (max_ar - min_ar)
        # compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        # check out-of-bounds
        h = min(h, self.seqlen_h - 1)
        w = min(w, self.seqlen_w - 1)

        return h, w

    def _sample_block_mask(self, block_size):
        block_h, block_w = block_size

        # sample bounding box
        top = self.rng.integers(0, self.seqlen_h - block_h)
        left = self.rng.integers(0, self.seqlen_w - block_w)
        bot = top + block_h
        right = left + block_w
        # create mask
        mask = torch.zeros((self.seqlen_h, self.seqlen_w), dtype=torch.int32)
        mask[top:bot, left:right] = 1
        mask = mask.flatten().nonzero().squeeze()
        # create complement
        mask_complement = torch.ones((self.seqlen_h, self.seqlen_w), dtype=torch.int32)
        mask_complement[top:bot, left:right] = 0
        return mask, mask_complement

    def _sample_block_mask_constrained(self, block_size, acceptable_regions):
        block_h, block_w = block_size

        tries = 0
        while True:
            # sample bounding box
            top = self.rng.integers(0, self.seqlen_h - block_h)
            left = self.rng.integers(0, self.seqlen_w - block_w)
            bot = top + block_h
            right = left + block_w
            # create mask
            mask = torch.zeros((self.seqlen_h, self.seqlen_w), dtype=torch.int32)
            mask[top:bot, left:right] = 1
            # constrain
            for k in range(max(int(len(acceptable_regions) - tries // self.tries), 0)):
                mask *= acceptable_regions[k]
            mask = mask.flatten().nonzero()
            if len(mask) > self.min_keep:
                break
            # increase tries (relax constraint every self.tries tries)
            tries += 1
        mask = mask.squeeze()
        return mask