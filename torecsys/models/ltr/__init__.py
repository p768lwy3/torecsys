"""torecsys.models.ltr is a sub module of the implementations of Learning-to-Rank algorithm"""

import torch.nn as nn

class _LtrModel(nn.Module):
    def __init__(self):
        super(_LtrModel, self).__init__()
