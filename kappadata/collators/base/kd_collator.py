class KDCollator:
    @property
    def default_collate_mode(self):
        raise NotImplementedError

    def __call__(self, *_, **__):
        raise RuntimeError("use collators in combination with KDCollator")

    def collate(self, batch, dataset_mode, ctx=None):
        raise NotImplementedError
