##

- batchwise denorm transforms
- factory

- unify prefixes (some objects start with KD others dont)

- cleanup seeds
- wrappers should also have set_rng instead of seed

- cleanup imports to always use the module instead of the concrete file

- collator rng

## features

- mixup/cutmix with binary label
- label smoothing for multiclass

## tests

## code

- modewrapper which excludes some kind of wrapper (e.g. exclude LabelSmoothingWrapper)
    - maybe some shallow copy solution (?)

- concatdataset handling of recursive stuff
