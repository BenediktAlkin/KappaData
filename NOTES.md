# Randomness

Randomness in Transforms/Collators is dependent on the number of workers (just like pytorch)

## Dataset wrappers

Dataset wrappers shouldn't need RNGs and in the case where they need it, random numbers can be immediately generated
which are then copied to the worker processes without requiring a RNG as member variable.

Examples:

- `wrappers.dataset_wrappers.ShuffleWrapper`

## Transforms/Collators

Transforms and Collators are ran in every dataloader worker process but may also be ran from the main process
(e.g. for loading a single sample to determine the input shape). To solve these two cases, each stochastic
Transform/Collator initializes a RNG in its constructor (using a seed that is sampled from the global RNG to make it
deterministic when setting `np.random.set_seed` as using `np.random.default_rng()` is not dependent on the global seed).
This RNG will then be copied into the worker processes which would result in the same RNG for all worker processes. As
this is not desired, each RNG is overwritten when the worker process is spawned via `worker_init_fn`. In
each `worker_init_fn` a new rng is created, again, by sampling a seed from the global RNG. This will result in different
RNGs for each worker due to the fact that the dataloader worker spawning process sets a different global seed per
worker.