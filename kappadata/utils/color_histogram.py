import torch

def color_histogram(images, bins, density=False):
    # image should be a batch of denormalized images
    assert images.ndim == 4
    assert torch.all(images >= 0.) and torch.all(images <= 255.)

    # create histogram
    # assign a bin to each pixel (values will be in [0, bins])
    pixel_bin_indices = (images.flatten(start_dim=2) / (256 // bins)).long()
    # calculate density of bins
    # https://stackoverflow.com/questions/69429586/how-to-get-a-histogram-of-pytorch-tensors-in-batches
    one_hot = torch.nn.functional.one_hot(pixel_bin_indices, bins)
    if density:
        return one_hot.float().mean(dim=-2)
    else:
        return one_hot.sum(dim=-2)