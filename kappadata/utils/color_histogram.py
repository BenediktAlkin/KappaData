import torch
import math

def color_histogram(images, bins, density=False, batch_size=None):
    # image should be a batch of denormalized images
    assert images.ndim == 4
    assert torch.all(images >= 0.) and torch.all(images <= 255.)
    assert 0 < bins <= 256
    assert 256 % bins == 0
    n_images, channels, height, width = images.shape

    # calculate in chunks to avoid OOM
    if batch_size is None:
        n_chunks = 1
    else:
        n_chunks = math.ceil(n_images / batch_size)
    counts = []
    for chunk in images.chunk(n_chunks):
        # create histogram
        # assign a bin to each pixel
        # shape: bs c h w -> bs c (h w)
        # range: [0, 255] -> [0, bins)
        chunk = (chunk.flatten(start_dim=2) / (256 // bins)).long()
        # count occourances of bins
        # https://stackoverflow.com/questions/69429586/how-to-get-a-histogram-of-pytorch-tensors-in-batches
        chunk = torch.nn.functional.one_hot(chunk, bins)
        counts.append(chunk.sum(dim=-2))
    counts = torch.concat(counts)

    if density:
        return counts / (height * width)
    return counts