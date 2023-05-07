# KappaData

[![publish](https://github.com/BenediktAlkin/KappaData/actions/workflows/publish.yaml/badge.svg)](https://github.com/BenediktAlkin/KappaData/actions/workflows/publish.yaml)

Utilities for [datasets and dataloading](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
with [pytorch](https://pytorch.org/)

- modular datasets
- caching datasets in-memory
- additional transforms
    - allow deterministic augmentations (e.g. for calculating the test loss on augmented samples)
    - [RandAugment](https://arxiv.org/abs/1909.13719)
    - patchwise augmentations
- various dataset filters and other dataset manipulation
    - filter by class
    - limit size to a %
    - [Mixup](https://arxiv.org/abs/1710.09412)
    - [Cutmix](https://arxiv.org/abs/1905.04899)
    - label smoothing
    - ...
- repeated augmentations

# Modular datasets

[pytorch datasets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) load all data in the `__getitem__`.
KappaData decouples the `__getitem__` such that single properties of the dataset can be loaded independently.

## Image classification dataset example

Let's take an image classification dataset as an example. A sample consists of an image with an associated class label.

```
class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        class_label = image_path_to_class_label(self.image_paths[idx])
        return img, class_label
```

If your training process contains something that only requires the class labels, the dataset has to additionally load
all the images which can take a long time (whereas loading only labels is very fast). With KappaData the `__getitem__`
method is split into subparts:

```
# inherit from kappadata.KDDataset
class ImageClassificationDataset(kappadata.KDDataset):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
    def __len__(self):
        return len(self.image_paths)
    
    # replace __getitem__ with getitem_x and getitem_y
    def getitem_x(self, idx, ctx=None):
        return load_image(self.image_paths[idx])
    def getitem_y(self, idx, ctx=None):
        return image_path_to_class_label(self.image_paths[idx])
```

Now each subpart of the dataset can be retrieved by wrapping the dataset into a `ModeWrapper`:

```
ds = ImageClassificationDataset(image_paths=...)
for y in kappadata.ModeWrapper(ds, mode="y"):
    ...
for x, y in kappadata.ModeWrapper(ds, mode="x y"):
    ...
```

[torch.utils.data.Subset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset) /
[torch.utils.data.ConcatDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset)
can be used by simply replacing them with `kappadata.KDSubset`/`kappadata.KDConcatDataset`.

# Wrappers

## "Dataset Wrappers"

KappaData implements various ways to manipulate datasets (`kappadata.wrappers.dataset_wrappers`).

- Filter by class
    - `kappadata.ClassFilterWrapper(ds, valid_classes=[0, 1])`
    - `kappadata.ClassFilterWrapper(ds, invalid_classes=[0, 1])`
- Balance data by oversampling underrepresented classes `kappadata.OversamplingWrapper(ds)`
- Subset by specifying percentages
    - `kappadata.PercentFilterWrapper(ds, from_percent=0.25)`
    - `kappadata.PercentFilterWrapper(ds, to_percent=0.75)`
    - `kappadata.PercentFilterWrapper(ds, from_percent=0.25, to_percent=0.75)`
- Repeat the whole dataset
    - repeat twice: `kappadata.RepeatWrapper(ds, repetitions=2)`
    - repeat until size is > 100 `kappadata.RepeatWrapper(ds, min_size=100)`
- Shuffle dataset
    - `kappadata.ShuffleWrapper(ds, seed=5)`

## "Sample Wrappers"

KappaData implements various ways to manipulate how samples are sampled from the underlying dataset
(`kappadata.wrappers.sample_wrappers`). "Sample Wrappers" are similar to transforms in that they transform the sample in
some way, but "Sample Wrappers" are more powerful because they have full access to the underlying dataset whereas normal
transforms only have access to a single sample.

```
class Transform:
  def forward(x):
    # only x can be manipulated (e.g. normalized, image-transforms, ...)
```

```
class SampleWrapper(kd.KDWrapper):
  def getitem_x(idx, ctx=None):
    # access to the underlying dataset via self.dataset
    # e.g. return the sum of two different samples
    idx2 = np.random.randint(len(self))
    return self.dataset.getitem_x(idx, ctx) + self.dataset.getitem_x(idx2, ctx)
```

This allows implementing more complex transformations. KappaData implements the following SampleWrappers:

- [Mixup](https://arxiv.org/abs/1710.09412) `kappadata.MixupWrapper(dataset=ds, alpha=1., p=1.)`
- [Cutmix](https://arxiv.org/abs/1905.04899) `kappadata.CutmixWrapper(dataset=ds, alpha=1., p=1.)`
- [Mixup](https://arxiv.org/abs/1710.09412) or [Cutmix](https://arxiv.org/abs/1905.04899)
  `kappadata.MixWrapper(dataset=ds, cutmix_alpha=1., mixup_alpha=1., p=1., cutmix_p=0.5)`
- TODO sampling multiple views
- label smoothing `kappadata.LabelSmoothingWrapper(dataset=ds, smoothing=.1)`

## Augmentation parameters

With KappaData you can also retrieve various properties of your data prepocessing (e.g. augmentation parameters). The
following example shows how you can retrieve the parameters
of [torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)
.

```
import torchvision.transforms.functional as F
class MyRandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def forward(self, img, ctx=None):
        # make random resized crop
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        cropped = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        # store parameters
        if ctx is not None:
          ctx["crop_parameters"] = (i, j, h, w)
        return cropped
  
class ImageClassificationDataset(kappadata.KDDataset):
    def __init__(self, ...):
      ...
      self.random_resized_crop = MyRandomResizedCrop()
    ...
    def getitem_x(self, idx, ctx=None):
        img = load_image(self.image_paths[idx])
        return self.random_resized_crop(img, ctx=ctx)
```

When you want to access the parameters simply pass `return_ctx=True` to the `ModeWrapper`:

```
ds = ImageClassificationDataset(image_paths=...)
for x, ctx in kappadata.ModeWrapper(ds, mode="x", return_ctx=True):
    print(ctx["crop_parameters"])
for (x, y), ctx in kappadata.ModeWrapper(ds, mode="x y", return_ctx=True):
    ...
```

# Caching datasets in-memory

## SharedDictDataset

`kappadata.SharedDictDataset` provides a wrapper to store arbitrary datasets in-memory via a dictionary shared between
all worker processes (using [python multiprocessing](https://docs.python.org/3/library/multiprocessing.html) data
structures). The shared memory part is important
for [dataloading](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
with `num_workers > 0`. Small and medium sized datasets can be cached in-memory to avoid bottlenecks when loading data
from a disk. For example even the full [ImageNet](https://www.image-net.org/) can be cached on many servers as it has ~
130GB and its not too uncommon for GPU servers to have more RAM than that.

## Caching image datasets

Naively caching image datasets can lead to high memory consumption because image data is usually stored in a compressed
format and decompressed during loading. To reduce memory, the raw uncompressed data needs to be cached.

Example caching
a [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html):

```
from kappadata.loading.image_folder import raw_image_loader, raw_image_folder_sample_to_pil_sample 
class CachedImageFolder(kappadata.KDDataset):
    def __init__(self, ...):
        # modify ImageFolder to load raw samples (NOTE: can't apply transforms onto raw data)
        self.ds = torchvision.datasets.ImageFolder(..., transform=None, loader=raw_image_loader)
        # initialize cached dataset that decompresses the raw data into a PIL image
        self.cached_ds = kappadata.SharedDictDataset(self.ds, transform=raw_image_folder_sample_to_pil_sample)
        # store transforms to apply after decompression
        self.transform = ...
    def getitem_x(self, idx, ctx=None):
        x, _ = self.cached_ds[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x
```

# Automatically copy datasets to a local (fast) disk

Datasets are often stored on a global (slow) storage and before training moved to a local (fast) disk.
`kappadata.copy_folder_from_global_to_local` provides an utility function to do this automatically:

- local path doesn't exist -> automatically copy from global to local
- local path exists -> do nothing
- local path exists but is incomplete -> clear directory and copy again

```
from pathlib import Path
from kappadata import copy_folder_from_global_to_local
global_path = Path("/system/data/ImageNet")
local_path = Path("/local/data")
# /system/data/ImageNet contains a 'train' and a 'val' folder -> copy whole dataset
copy_folder_from_global_to_local(global_path, local_path)
# copy only "train"
copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
```

The above code will also work (without modification) if `/system/data/ImageNet` contains only 2 zip files
`train.zip` and `val.zip`

# Miscellaneous

- all datasets derived from `kappadata.KDDataset` automatically support python slicing
    - `all_class_labels = ModeWrapper(ds, mode="y")[:]`
    - `all_class_labels = ModeWrapper(ds, mode="y")[5:-3:2]`
- all datasets derived from `kappadata.KDDataset` implement __iter__
  ```
  for y in ModeWrapper(ds, mode="y"):
      ...
  ```
- retrieve the original dataset without wrappers `ds.root_dataset`