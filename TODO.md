## tests

- test has_wrapper_type and all_wrappers

## code

- modewrapper which excludes some kind of wrapper (e.g. exclude MixupWrapper)
- think about a clean solution to avoid context overwriting (e.g. two mixup wrappers)
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