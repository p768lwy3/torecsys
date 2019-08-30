r"""torecsys.losses is a sub module of implementation of losses in recommendation system.
"""

import torch.nn as nn

class _Loss(nn.Module):
    r"""General Loss class.
    """
    def __init__(self):
        # refer to parent class
        super(_Loss, self).__init__()

from .ctr import *
from .emb import *
from .ltr import * 
