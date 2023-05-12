from .kd_ade20k_norm import KDAde20kNorm
from .kd_cifar100_norm import KDCifar100Norm
from .kd_cifar10_norm import KDCifar10Norm
from .kd_image_net_norm import KDImageNetNorm


def string_to_norm(name):
    assert isinstance(name, str)
    name = name.lower().replace("_")
    if name == "cifar10":
        return KDCifar10Norm()
    if name == "cifar100":
        return KDCifar100Norm()
    if name == "imagenet":
        return KDImageNetNorm()
    raise NotImplementedError
