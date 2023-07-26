import einops
from torchvision.transforms.functional import to_pil_image
import math

from multiprocessing import Value

import torch

from kappadata.utils.param_checking import to_2tuple

class MaskCollatorOriginal:
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False
    ):
        super().__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    print(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return batch, collated_masks_enc, collated_masks_pred

class MaskCollator:

    def __init__(
            self,
            input_size=(224, 224),
            patch_size=16,
            enc_mask_scale=(0.85, 1.0),
            pred_mask_scale=(0.15, 0.2),
            predictor_aspect_ratio=(0.75, 1.5),
            num_enc_masks=1,
            num_pred_masks=4,
            min_keep=10,
            tries=20,
    ):
        super(MaskCollator, self).__init__()
        self.input_size = to_2tuple(input_size)
        self.patch_size = to_2tuple(patch_size)
        self.seqlen_h = self.input_size[0] // self.patch_size[0]
        self.seqlen_w = self.input_size[1] // self.patch_size[1]
        self.encoder_mask_scale = to_2tuple(enc_mask_scale)
        self.predictor_mask_scale = to_2tuple(pred_mask_scale)
        self.predictor_aspect_ratio = to_2tuple(predictor_aspect_ratio)
        self.num_enc_masks = num_enc_masks
        self.num_pred_masks = num_pred_masks
        self.min_keep = min_keep
        self.tries = tries
        self._itr_counter = Value('i', -1)

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
        # sample block aspect-ratio
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
        top = torch.randint(0, self.seqlen_h - block_h, size=(1,))
        left = torch.randint(0, self.seqlen_w - block_w, size=(1,))
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
            top = torch.randint(0, self.seqlen_h - block_h, size=(1,))
            left = torch.randint(0, self.seqlen_w - block_w, size=(1,))
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

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        batch_size = len(batch)

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

        masks_predictor, masks_encoder = [], []
        min_keep_pred = min_keep_enc = self.seqlen_h * self.seqlen_w
        for _ in range(batch_size):

            masks_pred, masks_pred_complement = [], []
            for _ in range(self.num_pred_masks):
                mask, mask_complement = self._sample_block_mask(predictor_size)
                masks_pred.append(mask)
                masks_pred_complement.append(mask_complement)
                min_keep_pred = min(min_keep_pred, len(mask))
            masks_predictor.append(masks_pred)

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

        return batch, masks_encoder, masks_predictor

def main():
    mask_collator_og = MaskCollatorOriginal(patch_size=14)
    mask_collator = MaskCollator(patch_size=14)
    batch = torch.randn(4, 3, 32, 32)
    #mask_collator_og(batch)
    _, masks_encoder, masks_predictor = mask_collator(batch)

    x_encoder = einops.rearrange(torch.zeros(4, 3, 16, 16), "b c h w -> b c (h w)")
    x_predictors = einops.rearrange(torch.zeros(4, 4, 3, 16, 16), "p b c h w -> p b c (h w)")
    gray = torch.tensor([0.2, 0.2, 0.2])
    for masks_enc in masks_encoder:
        for i, mask in enumerate(masks_enc):
            for idx in mask:
                x_encoder[i, :, idx] = gray

    for i, masks_pred in enumerate(masks_predictor):
        for j, mask in enumerate(masks_pred):
            for idx in mask:
                x_predictors[i, j, :, idx] = gray


    for i in range(len(x_encoder)):
        to_pil_image(einops.rearrange(x_encoder[i], "c (h w) -> c h w", h=16, w=16)).save(f"{i}_encoder.png")
    for j in range(len(x_predictors)):
        for i in range(len(x_predictors[j])):
            to_pil_image(einops.rearrange(x_predictors[j][i], "c (h w) -> c h w", h=16, w=16)).save(f"{i}_pred{j}.png")

    print("fin")



if __name__ == "__main__":
    main()
