from torch.utils.data import SequentialSampler as TorchSequentialSampler


class SequentialSampler(TorchSequentialSampler):
    @property
    def effective_length(self):
        return len(self)
