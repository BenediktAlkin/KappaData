##

- seed for augmentations per epochs (generate a testset with augmentations)
- add retrieving via ctx. to readme

## features

- look at differences to timm mixes
- collate mixup/cutmix tests
- mixup/cutmix with binary label

## other

- check why some torchvision.transforms are nn.Module and some are not

## tests

- test mixup collators
- mixup with p
- all mix wrappers: automatic tests with and without context information
- xtransformwrapper tests

## code

- MultiViewWrapper
- modewrapper which excludes some kind of wrapper (e.g. exclude LabelSmoothingWrapper)
    - maybe some shallow copy solution (?)

- concatdataset handling of recursive stuff
- check if copying folders is faster than individual files
- test ClassFilterWrapper with string labels
- check how test requirements are correctly handled
    - currently `# noinspection PyPackageRequirements` is needed for pyfakefs

## NOTES

- transforms cant use torch rng `TypeError: cannot pickle 'torch._C.Generator' object` -> use np for random generator

## caching

pip install redis[hiredis]

### redis improvements

- multiple redis instances
    - store guid in database and check for it

## other

- inmemory zip
- samplers