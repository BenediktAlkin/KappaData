import kappadata.collators as kdc

class MAEFinetuneMixCollator(kdc.KDMixCollator):
    def __init__(self):
        super().__init__(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            mixup_p=0.5,
            cutmix_p=0.5,
            apply_mode="batch",
            lamb_mode="batch",
            shuffle_mode="flip",
        )