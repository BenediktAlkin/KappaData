## features

- collate integration
- collate mixup/cutmix
- mixup/cutmix with binary label

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