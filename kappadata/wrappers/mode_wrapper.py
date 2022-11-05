from kappadata.datasets.kd_dataset import KDDataset


class ModeWrapper(KDDataset):
    def __init__(self, dataset: KDDataset, mode: str, return_ctx: bool = False):
        super().__init__()
        self.dataset = dataset
        self.mode = mode
        self.return_ctx = return_ctx

        self._getitem_fns = []
        items = mode.split(" ")
        for item in items:
            if item == "index":
                self._getitem_fns.append(self._getitem_index)
            else:
                fn_name = f"getitem_{item}"
                assert hasattr(self.dataset, fn_name), f"{type(self.dataset.root_dataset)} has no method getitem_{item}"
                self._getitem_fns.append(getattr(self.dataset, fn_name))

    @staticmethod
    def _getitem_index(idx, _=None):
        return idx

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(len(self))[idx]]
        if isinstance(idx, list):
            return [self[i] for i in idx]
        if idx < 0:
            idx = len(self) + idx

        items = []
        ctx = {}
        for getitem_fn in self._getitem_fns:
            item = getitem_fn(idx, ctx)
            items.append(item)
        if len(items) == 1:
            # single item -> no tuple
            items = items[0]
        else:
            # multiple items -> wrap into tuple
            items = tuple(items)
        if self.return_ctx:
            return items, ctx
        return items

    def __len__(self):
        return len(self.dataset)

    @property
    def root_dataset(self):
        # root_dataset is implemented in base class -> not handled in __getattr__
        return self.dataset.root_dataset

    def has_wrapper(self, wrapper):
        if self == wrapper:
            return True
        return self.dataset.has_wrapper(wrapper)

    def has_wrapper_type(self, wrapper_type):
        if type(self) == wrapper_type:
            return True
        return self.dataset.has_wrapper_type(wrapper_type)

    @property
    def all_wrappers(self):
        return [self] + self.dataset.all_wrappers

    @property
    def all_wrapper_types(self):
        return [type(self)] + self.dataset.all_wrapper_types

    def __getattr__(self, item):
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    def __iter__(self):
        """ torch.utils.data.Dataset doesn't define __iter__ which makes 'for sample in dataset' run endlessly """
        for i in range(len(self)):
            yield self[i]
