class CollatorBase:
    def __init__(self, dataset_mode):
        self.dataset_mode = dataset_mode

    def __call__(self, *_, **__):
        raise RuntimeError("use collators in combination with KDCollator")

    def collate(self, batch, ctx=None):
        raise NotImplementedError
