from kappadata.common.transforms.byol_transforms import BYOLTransform0, BYOLTransform1
from kappadata.wrappers.sample_wrappers.multi_view_wrapper import MultiViewWrapper


class BYOLMultiViewWrapper(MultiViewWrapper):
    def __init__(self, dataset):
        super().__init__(dataset=dataset, configs=[BYOLTransform0(), BYOLTransform1()])
