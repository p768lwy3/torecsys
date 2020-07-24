r"""[In development] torecsys.estimators is a sub module of estimators, which can be called directly with a fitted dataloader of inputs
"""

import warnings

from torecsys.trainer import Trainer

# ignore import warnings of the below packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # from tensorboardX import SummaryWriter


class _Estimator(Trainer):
    r"""Base class of estimator provide several function would be used in training and 
    inference."""

    def __init__(self, **kwargs):
        super(_Estimator, self).__init__(**kwargs)


from .ctr import *
from .emb import *
from .ltr import *
