from dataclasses import dataclass
from kappadata.transforms.base.kd_transform import KDTransform
from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform

@dataclass
class IsDeterministicTransformResult:
    deterministic_transforms: list
    nondeterministic_transforms: list
    unknown_transforms: list

    @property
    def is_deterministic(self):
        return (
                len(self.unknown_transforms) == 0 and
                len(self.nondeterministic_transforms) == 0 and
                self.is_randomly_seeded
        )

    @property
    def all_kd_transforms_are_deterministic(self):
        return len(self.nondeterministic_transforms) == 0 and self.is_randomly_seeded

    @property
    def is_randomly_seeded(self):
        return is_randomly_seeded_transform(self.deterministic_transforms)

def _transform_to_transforms(transform):
    # import is here due to circular dependency
    from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
    if isinstance(transform, list):
        return transform
    if isinstance(transform, KDComposeTransform):
        return transform.transforms
    return [transform]

def is_randomly_seeded_transform(transform):
    transforms = _transform_to_transforms(transform)
    seeds = [t.seed for t in transforms if isinstance(t, KDStochasticTransform) and t.seed is not None]
    return len(seeds) == len(set(seeds))

def has_stochastic_transform_with_seed(transform):
    transforms = _transform_to_transforms(transform)
    return any(t.seed is not None for t in transforms if isinstance(t, KDStochasticTransform))

def is_deterministic_transform(transform) -> IsDeterministicTransformResult:
    transforms = _transform_to_transforms(transform)

    deterministic_transforms = []
    nondeterministic_transforms = []
    unknown_transforms = []
    for transform in transforms:
        if isinstance(transform, KDStochasticTransform):
            if transform.seed is not None:
                deterministic_transforms.append(transform)
            else:
                nondeterministic_transforms.append(transform)
        elif isinstance(transform, KDTransform):
            deterministic_transforms.append(transform)
        else:
            unknown_transforms.append(transform)

    return IsDeterministicTransformResult(
        deterministic_transforms=deterministic_transforms,
        nondeterministic_transforms=nondeterministic_transforms,
        unknown_transforms=unknown_transforms,
    )