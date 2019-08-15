r"""shorcut to call torecsys.models.*.estimators.* classes and functions
"""

import torch.nn as nn

class _Estimator(nn.Module):
    def __init__(self):
        super(_Estimator, self).__init__()
    
    def size(self):
        return self.nparams

from .models.ctr.estimators import *
from .models.emb.estimators import *
from .models.ltr.estimators import *
