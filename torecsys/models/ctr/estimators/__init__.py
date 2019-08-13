r"""torecsys.models.ctr.estimators is a sub module of the implementation of several click through rate models, which can be called directly
"""

import torch
import torch.nn as nn

class _CtrEstimator(nn.Module):
    def __init__(self):
        super(_CtrEstimator, self).__init__()

from .factorization_machine import FactorizationMachine
