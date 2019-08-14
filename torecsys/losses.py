r"""shorcut to call torecsys.models.*.losses.* classes and functions
"""

import torch.nn as nn

class _Loss(nn.Module):
    def __init__(self):
        super(_Loss, self).__init__()

# from .models.ctr.losses import *
from .models.emb.losses import *
from .models.ltr.losses import *
