##

- test _collate_batchwise
- look at the mixup/cumix implementations -> can be made much cleaner (make a functional, call functional from sample_wrappers and collators)
- https://pytorch.org/blog/extending-torchvisions-transforms-to-object-detection-segmentation-and-video-tasks/?utm_source=twitter&utm_medium=organic_social&utm_campaign=evergreen
- make mixup/cutmix more performant with
- also the label classes seem useful

## features

- ComposeTransform for KDTransform
- look at differences to timm mixes
- collate mixup/cutmix tests
- mixup/cutmix with binary label

## other

- check why some torchvision.transforms are nn.Module and some are not

## tests

- mixup with p
- all mix wrappers: automatic tests with and without context information
- xtransformwrapper tests

## code

- think about a clean solution to avoid context overwriting (e.g. two mixup wrappers)
- MultiViewWrapper
- modewrapper which excludes some kind of wrapper (e.g. exclude MixupWrapper)
    - maybe some shallow copy solution (?)

- concatdataset handling of recursive stuff
- check if copying folders is faster than individual files
- test ClassFilterWrapper with string labels
- check how test requirements are correctly handled
    - currently `# noinspection PyPackageRequirements` is needed for pyfakefs

## caching

pip install redis[hiredis]

### redis improvements

- multiple redis instances
    - store guid in database and check for it

## other

- inmemory zip
- samplers