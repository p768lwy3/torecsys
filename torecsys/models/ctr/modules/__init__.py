r"""torecsys.models.ctr.modules is a sub module of the implementation of modules in severals models, which can be concatenate to CTR model
"""

import torch
import torch.nn as nn

class _CtrModule():
    def __init__(self):
        super(_CtrModule, self).__init__()
    
    def nparams(self):
        return self.nparams

from .factorization_machine import FactorizationMachine
