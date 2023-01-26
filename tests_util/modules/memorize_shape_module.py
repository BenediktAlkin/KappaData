import torch.nn as nn


class MemorizeShapeModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape))
        return self.module(x)
