

class InterleavedBatchSampler:
    def __init__(self, interleaved_sampler):
        super().__init__()
        self.interleaved_sampler = interleaved_sampler

    def __iter__(self):
        idxs = []
        for is_full_batch, idx in self.interleaved_sampler:
            idxs.append(idx)
            if is_full_batch:
                yield idxs
                idxs = []
        assert len(idxs) == 0


    def __len__(self):
        raise NotImplementedError
