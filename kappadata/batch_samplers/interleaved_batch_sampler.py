

class InterleavedBatchSampler:
    def __init__(self, interleaved_sampler):
        super().__init__()
        self.interleaved_sampler = interleaved_sampler

    def __iter__(self):
        idxs = []
        for is_full_batch, idx in self.interleaved_sampler:
            if is_full_batch:
                yield idxs
                idxs = []
            else:
                idxs.append(idx)

    def __len__(self):
        raise NotImplementedError
