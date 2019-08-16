r"""torecsys.models is a sub module of implementation with a whole models in recommendation system
"""

import torch.nn as nn

class _Model(nn.Module):
    def __init__(self):
        super(_Model, self).__init__()

from .ctr import *
from .emb import *
from .ltr import *
