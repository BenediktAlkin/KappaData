from kappadata.wrappers.sample_wrappers.multi_view_wrapper import MultiViewWrapper
from kappadata.common.transforms.byol_transforms import BYOLTransform0, BYOLTransform1

class BYOLMultiViewWrapper(MultiViewWrapper):
    def __init__(self):
        super().__init__(configs=[BYOLTransform0(), BYOLTransform1()])