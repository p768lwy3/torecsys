from . import _CtrModel
from torecsys.layers import WideLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn

class LogisticRegression(_CtrModel):
    def __init__(self):
        self.linear = nn.Linear()
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        return 