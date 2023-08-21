import einops

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.one_hot import to_one_hot_vector
import numpy as np
import torch


class KDRandomClassWrapper(KDWrapper):
    def __init__(self, dataset, mode="random", mode_kwargs=None, num_classes=None, seed=0, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        if num_classes is None:
            class_shape = self.dataset.getshape("class")
            assert len(class_shape) == 1
            self._num_classes = class_shape[0]
        else:
            self._num_classes = num_classes
        self._seed = seed
        self._mode = mode
        self._mode_kwargs = mode_kwargs
        self._generate_classes()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        self._generate_classes()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
        self._generate_classes()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._generate_classes()

    @property
    def class_names(self):
        raise NotImplementedError

    def _generate_classes(self):
        size = len(self.dataset)
        generator = torch.Generator().manual_seed(self._seed)
        if self._mode == "random":
            gen_fn = self._random
        elif self._mode == "randperm":
            gen_fn = self._randperm
        elif self._mode == "gatherbug":
            gen_fn = self._gatherbug
        else:
            raise NotImplementedError
        kwargs = dict(num_classes=self._num_classes, size=size, generator=generator)
        self._classes = gen_fn(**kwargs, **self._mode_kwargs or {}).tolist()

    @staticmethod
    def _random(num_classes, size, generator):
        return torch.randint(num_classes, size=(size,), generator=generator)

    @staticmethod
    def _randperm(num_classes, size, generator):
        return KDRandomClassWrapper.__repeat(torch.randperm(num_classes, generator=generator), size, num_classes)

    # noinspection PyUnusedLocal
    @staticmethod
    def _gatherbug(num_classes, size, world_size, **_):
        # generate original classes
        samples_per_class = (size + num_classes - 1) // num_classes
        classes = torch.arange(num_classes).repeat_interleave(samples_per_class)[:size]
        # pad just like DistributedSampler would pad
        num_padded_samples = (world_size - size % world_size) % world_size
        if num_padded_samples > 0:
            classes = torch.concat([classes, classes[:num_padded_samples]])
        # revert order DistributedSampler split [0, 1, 2, 3, 4, 5, 6, 7] with 4 GPUs into [0, 4, 1, 5, 2, 6, 3, 7]
        classes = einops.rearrange(classes, "(classes world_size) -> (world_size classes)", world_size=world_size)
        # remove padded samples
        if num_padded_samples > 0:
            classes = classes[:size]
        return classes

    @staticmethod
    def __repeat(tensor, size, num_classes):
        return tensor.repeat((size + num_classes - 1) // num_classes)[:size]

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self._classes[idx]

    def getall_class(self):
        return self._classes

    def getshape_class(self):
        return self._num_classes,