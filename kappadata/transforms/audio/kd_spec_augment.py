from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform


class KDSpecAugment(KDStochasticTransform):
    def __init__(self, frequency_masking=None, time_masking=None, **kwargs):
        super().__init__(**kwargs)
        assert (frequency_masking is not None) or (time_masking is not None)
        self.frequency_masking = frequency_masking
        self.time_masking = time_masking

    def __call__(self, x, ctx=None):
        assert x.ndim == 3
        if self.frequency_masking is not None:
            self._mask_along_axis(
                specgram=x,
                mask_param=self.frequency_masking,
                axis=1,
            )
        if self.time_masking is not None:
            self._mask_along_axis(
                specgram=x,
                mask_param=self.time_masking,
                axis=2,
            )
        return x

    def _mask_along_axis(
            self,
            specgram,
            mask_param: int,
            axis: int,
            p: float = 1.0,
            fill_value: float = 0.0,
    ):
        """ torchaudio.functional.functional.mask_along_axis but numpy rng """

        if axis not in [1, 2]:
            raise ValueError("Only Frequency and Time masking are supported")

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

        if p < 1.0:
            mask_param = min(mask_param, int(specgram.shape[axis] * p))

        if mask_param < 1:
            return specgram

        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape([-1] + list(shape[-2:]))
        value = torch.tensor(self.rng.random()) * mask_param
        min_value = torch.tensor(self.rng.random()) * (specgram.size(axis) - value)

        mask_start = (min_value.long()).squeeze()
        mask_end = (min_value.long() + value.long()).squeeze()
        mask = torch.arange(0, specgram.shape[axis], device=specgram.device, dtype=specgram.dtype)
        mask = (mask >= mask_start) & (mask < mask_end)
        if axis == 1:
            mask = mask.unsqueeze(-1)

        assert mask_end - mask_start < mask_param

        specgram = specgram.masked_fill(mask, fill_value)

        # unpack batch
        specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

        return specgram