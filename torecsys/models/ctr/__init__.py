"""torecsys.models.ctr is a sub module of the implementations of Click Through Rate (ctr) Prediction's algorithm"""

import torch.nn as nn

class _CtrModel(nn.Module):
    def __init__(self):
        super(_CtrModel, self).__init__()
