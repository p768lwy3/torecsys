r"""torecsys.models.ltr.estimators is a sub module of the implementation of several learning-to-rank models, which can be called directly
"""

from torecsys.models import _Estimator
import torch
import torch.nn as nn

class _LtrEstimator(_Estimator):
    def __init__(self):
        super(_LtrEstimator, self).__init__()
        