import math
from dataclasses import dataclass

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.transforms.functional import get_image_size
from tqdm import tqdm


@dataclass
class HistogramData:
    values: list
    start: int
    end: int
    avg: float


def _noop_collate(batch):
    return batch


def _to_histogram(tensor, cutoff, bucket_size, batch_size):
    # calculate in chunks to avoid OOM)
    counts = []
    tensor_min = tensor.min()
    bins = ((tensor.max() - tensor_min) / bucket_size + 1).long().item()
    tensor = tensor - tensor_min
    for chunk in tqdm(tensor.chunk(math.ceil(len(tensor) / batch_size))):
        # count occourances of bins
        # https://stackoverflow.com/questions/69429586/how-to-get-a-histogram-of-pytorch-tensors-in-batches
        chunk = one_hot(chunk.div(bucket_size, rounding_mode="trunc"), bins)
        counts.append(chunk.sum(dim=0))
    # average counts over chunks
    counts = torch.stack(counts).float().mean(dim=0)
    # convert to density
    counts /= counts.sum()

    # cut off outliers
    cumsum = torch.cumsum(counts, dim=0)
    start_idx = (cumsum >= cutoff).nonzero().min()
    end_idx = (cumsum <= (1 - cutoff)).nonzero().max()
    counts = counts[start_idx:end_idx]
    # renormalize
    counts /= counts.sum()

    return counts.cpu(), (tensor_min + start_idx * bucket_size).cpu(), (tensor_min + (end_idx - 1) * bucket_size).cpu()


def visualize_dataset_imgsize(dataset, cutoff=0.05, bucket_size=1, batch_size=128, num_workers=10, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_noop_collate)

    heights = []
    widths = []
    for batch in tqdm(dataloader):
        for img in batch:
            width, height = get_image_size(img)
            heights.append(height)
            widths.append(width)
    heights = torch.tensor(heights)
    widths = torch.tensor(widths)

    device = torch.device(device)
    heights = heights.to(device)
    widths = widths.to(device)

    areas = heights * widths
    kwargs = dict(cutoff=cutoff, bucket_size=bucket_size, batch_size=batch_size)
    hist_heights, heights_min, heights_max = _to_histogram(heights, **kwargs)
    hist_widths, widths_min, widths_max = _to_histogram(widths, **kwargs)
    hist_areas, areas_min, areas_max = _to_histogram(areas, **kwargs)

    return (
        HistogramData(values=hist_heights, start=heights_min, end=heights_max, avg=heights.float().mean().cpu()),
        HistogramData(values=hist_widths, start=widths_min, end=widths_max, avg=widths.float().mean().cpu()),
        HistogramData(values=hist_areas, start=areas_min, end=areas_max, avg=areas.float().mean().cpu()),
    )
