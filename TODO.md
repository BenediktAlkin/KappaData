##

- collators cant be used as part of deterministically augmented dataset

- copy folder tests: `copy_folder_from_global_to_local("~/Documents/data/coco", "~/Documents/data_local/coco", relative_path="val2017", log_fn=print)`

- worker_init_fn for collator + call it (to initialize rng per worker process) 
- refactor to import submodules (e.g. from kappdata.transforms import ... instead of from kappadata import ...)

- skip forward in interleaved sampler
- port everything to use torch operations (much better asymptotic complexity for e.g. oversampling wrapper)
- variable batch_size in interleaved sampler (batch_size per config)
- get_wrapper_of_type doesnt consider inherited types (e.g. for BYOL augmentation wrapper)
- refactor everything to use from .module import (class1, class2, ...)
- distributed sampler with any sampler (not just RandomSampler/SequentialSampler)
- test interleaved sampler
- test batchwise denormalization
- normalizations can be implemented much cleaner by merging sample and batchwise operations together
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
