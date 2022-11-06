class KDCollator:
    def __init__(self, dataset_mode):
        self.dataset_mode = dataset_mode

    @property
    def default_collate_mode(self):
        raise NotImplementedError

    def __call__(self, *_, **__):
        raise RuntimeError("use collators in combination with KDCollator")

    def collate(self, batch, ctx=None):
        raise NotImplementedError