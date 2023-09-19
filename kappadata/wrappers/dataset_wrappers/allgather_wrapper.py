import torch

from kappadata.datasets.kd_subset import KDSubset
from kappadata.utils.getall_class_as_tensor import getall_class_as_tensor
import einops

class AllgatherWrapper(KDSubset):
    def __init__(self, dataset, world_size):
        indices = torch.arange(len(dataset))
        num_padded_samples = (world_size - len(indices) % world_size) % world_size

        # pad to multiple of world size
        if num_padded_samples > 0:
            indices = torch.concat([indices, indices[:num_padded_samples]])
        # rearrange
        indices = einops.rearrange(
            tensor=indices,
            pattern="(world_size samples_per_gpu) -> (samples_per_gpu world_size)",
            world_size=world_size
        )
        # cut away padding
        if num_padded_samples > 0:
            indices = indices[:-num_padded_samples]
        super().__init__(dataset=dataset, indices=indices.tolist())
