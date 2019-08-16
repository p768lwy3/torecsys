r"""torecsys.losses is a sub module of implementation of losses used in any kind of models
"""

import torch.nn as nn

class _Loss(nn.Module):
    def __init__(self):
        super(_Loss, self).__init__()

from .ctr import *
from .emb import *
from .ltr import * 
