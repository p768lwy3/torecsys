r"""shortcut to call torecsys.models.*.modules.* classes and functions
"""

import torch.nn as nn

class _Module(nn.Module):
    def __init__(self):
        super(_Module, self).__init__()
    
    def size(self):
        return self.nparams

from .models.ctr.modules import *
from .models.emb.modules import *
from .models.ltr.modules import *
