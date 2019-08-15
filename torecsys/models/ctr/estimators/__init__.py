r"""torecsys.models.ctr.estimators is a sub module of the implementation of several click through rate models, which can be called directly
"""

from torecsys.estimators import _Estimator
import torch.nn as nn

class _CtrEstimator(_Estimator):
    def __init__(self):
        super(_CtrEstimator, self).__init__()

from .factorization_machine import FactorizationMachineEstimator
