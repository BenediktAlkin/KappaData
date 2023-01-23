from kappadata.common.transforms.byol_transforms import BYOLTransform0, BYOLTransform1
from kappadata.wrappers.sample_wrappers.kd_multi_view_wrapper import KDMultiViewWrapper


class BYOLMultiViewWrapper(KDMultiViewWrapper):
    def __init__(self, dataset):
        super().__init__(dataset=dataset, configs=[BYOLTransform0(), BYOLTransform1()])
