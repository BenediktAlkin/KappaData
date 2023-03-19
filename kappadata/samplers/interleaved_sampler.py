from dataclasses import dataclass
from dataclasses import dataclass

from torch.utils.data import ConcatDataset, DistributedSampler


@dataclass
class InterleavedSamplerConfig:
    sampler: object
    every_n_epochs: int = None
    every_n_updates: int = None
    every_n_samples: int = None


class InterleavedSampler:
    def __init__(self, main_sampler, batch_size, configs=None, drop_last=True, epochs=None, updates=None, samples=None):
        super().__init__()
        assert isinstance(batch_size, int) and 0 < batch_size
        assert epochs is None or (isinstance(epochs, int) and 0 < epochs)
        assert updates is None or (isinstance(updates, int) and 0 < updates)
        assert samples is None or (isinstance(samples, int) and 0 < samples)
        assert sum([epochs is not None, updates is not None, samples is not None]) <= 1
        configs = configs or []
        for config in configs:
            assert (
                    (config.every_n_epochs is not None) or
                    (config.every_n_updates is not None) or
                    (config.every_n_samples is not None)
            )
            assert config.every_n_epochs is None or 0 < config.every_n_epochs
            assert config.every_n_updates is None or 0 < config.every_n_updates
            assert config.every_n_samples is None or 0 < config.every_n_samples

        self.main_sampler = main_sampler
        self.drop_last = drop_last
        self.configs = configs
        self.batch_size = batch_size
        self.epochs = epochs
        self.updates = updates
        self.samples = samples

        def _get_data_source(sampler):
            if hasattr(sampler, "data_source"):
                return sampler.data_source
            if hasattr(sampler, "dataset"):
                return sampler.dataset
            raise NotImplementedError

        self.index_offsets = [len(_get_data_source(self.main_sampler))]
        for config in self.configs[:-1]:
            self.index_offsets.append(self.index_offsets[-1] + len(_get_data_source(config.sampler)))
        self.dataset = ConcatDataset(
            [_get_data_source(self.main_sampler)] +
            [_get_data_source(config.sampler) for config in self.configs]
        )

    def __iter__(self):
        if self.drop_last:
            if len(self.main_sampler) < self.batch_size:
                self.batch_size = len(self.main_sampler)
            samples_per_epoch = len(self.main_sampler) // self.batch_size * self.batch_size
        else:
            samples_per_epoch = len(self.main_sampler)

        sample = 0
        epoch = 0
        update = 0
        sample_in_update = 0
        while True:
            sample_in_epoch = 0
            if isinstance(self.main_sampler, DistributedSampler):
                self.main_sampler.set_epoch(epoch)
            for main_idx in self.main_sampler:
                sample += 1
                sample_in_epoch += 1
                sample_in_update += 1
                if sample_in_update == self.batch_size or sample_in_epoch == samples_per_epoch:
                    yield True, main_idx
                else:
                    yield False, main_idx
                # check if interleaved dataset has to be iterated (only possible after a update)
                # sample_in_update == self.batch_size -> full batch
                # if not drop_last -> last batch is not full but is also an update
                if sample_in_update == self.batch_size or sample_in_epoch == samples_per_epoch:
                    # keep track of what the sample counter was at the last update for every_n_sample checks
                    sample_in_update = 0
                    sample_at_last_update = sample
                    # increase counters
                    update += 1
                    if sample_in_epoch == samples_per_epoch:
                        epoch += 1

                    # check if interleaved dataset has to be iterated
                    for config_idx, config in enumerate(self.configs):
                        if (
                                (config.every_n_epochs is not None and sample_in_epoch == samples_per_epoch) or
                                (config.every_n_updates is not None and update % config.every_n_updates == 0) or
                                (config.every_n_samples is not None and
                                 sample_at_last_update // config.every_n_samples < sample // config.every_n_samples)
                        ):
                            index_offset = self.index_offsets[config_idx]
                            sample_in_interleaved = 0
                            for interleaved_idx in config.sampler:
                                sample_in_interleaved += 1
                                if (
                                        sample_in_interleaved % self.batch_size == 0 or
                                        sample_in_interleaved == len(config.sampler)
                                ):
                                    yield True, index_offset + interleaved_idx
                                else:
                                    yield False, index_offset + interleaved_idx

                    # check if end is reached
                    if (
                            (self.epochs is not None and epoch == self.epochs) or
                            (self.updates is not None and update == self.updates) or
                            (self.samples is not None and sample >= self.samples)
                    ):
                        return
                    # if drop_last -> skip last non-full batch
                    if sample_in_epoch == samples_per_epoch:
                        break
