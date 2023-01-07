##

- new multiviewwrapper reset_seed; assert unequal seeds of all sub transforms


- context of random transform cant be stacked by torch collator as they only write values when they are applied
- label smoothing for multiclass
- seed for augmentations per epochs (generate a testset with augmentations)
- add retrieving via ctx. to readme

## features

- mixup/cutmix with binary label

## other

- check why some torchvision.transforms are nn.Module and some are not

## tests

## code

- MultiViewWrapper
- modewrapper which excludes some kind of wrapper (e.g. exclude LabelSmoothingWrapper)
    - maybe some shallow copy solution (?)

- concatdataset handling of recursive stuff
- test ClassFilterWrapper with string labels

## NOTES

- transforms cant use torch rng `TypeError: cannot pickle 'torch._C.Generator' object` -> use np for random generator
