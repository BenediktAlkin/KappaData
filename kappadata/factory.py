import inspect
from copy import deepcopy
from itertools import chain

import torchvision.transforms

_REGISTERED_TRANSFORMS = {}


def register_transform(name, cls):
    _REGISTERED_TRANSFORMS[name] = cls


def object_to_transform(obj):
    if obj is None:
        return None
    if not isinstance(obj, (list, dict)):
        return obj

    # implicit KDComposeTransform (list without any kind parameter)
    if isinstance(obj, list):
        transforms = []
        for transform in obj:
            transforms.append(object_to_transform(transform))
        from .transforms import KDComposeTransform
        return KDComposeTransform(transforms)

    # transform is explicitly defined via kind
    assert "kind" in obj and isinstance(obj["kind"], str)
    obj = deepcopy(obj)
    kind = obj.pop("kind")

    if kind in _REGISTERED_TRANSFORMS:
        return _REGISTERED_TRANSFORMS[kind](**obj)

    # import here to avoid circular dependencies
    import kappadata.common.transforms
    # get all names and ctors of possible transforms
    kd_pascal_ctor_list = inspect.getmembers(kappadata.transforms, inspect.isclass)
    kd_common_pascal_ctor_list = inspect.getmembers(kappadata.common.transforms, inspect.isclass)
    tv_pascal_ctor_list = inspect.getmembers(torchvision.transforms, inspect.isclass)
    # if duplicates occour the latest in the chain dominate
    pascal_to_ctor = {
        name: ctor
        for name, ctor in chain(
            tv_pascal_ctor_list,
            kd_pascal_ctor_list,
            kd_common_pascal_ctor_list,
        )
    }
    # allow snake_case (type name is PascalCase)
    if kind[0].islower():
        kind = kind.replace("_", "")
        snake_to_pascal = {name.lower(): name for name in pascal_to_ctor.keys()}
        assert kind in snake_to_pascal.keys(), f"invalid kind '{kind}' (possibilities: {snake_to_pascal.keys()})"
        kind = snake_to_pascal[kind]
    ctor = pascal_to_ctor[kind]
    return ctor(**obj)
