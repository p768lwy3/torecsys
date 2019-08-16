r"""torecsys.estimators.ltr is a sub module of the estimators of learning-to-rank models
"""

from .. import _Estimator

class _LtrEstimator(_Estimator):
    def __init__(self):
        super(_LtrEstimator, self).__init__()

class _LtrEstimator(_Estimator):
    r"""Base class of embedding estimator provide several functions would be called"""
    def __init__(self, 
                 model    : Callable,
                 loss     : Callable,
                 optimizer: Callable, 
                 epochs   : int,
                 verbose  : int,
                 logdir   : logdir = None):
        
        super(_LtrEstimator, self).__init__(
            model     = model,
            loss      = loss,
            optimizer = optimizer,
            epochs    = epochs,
            verbose   = verbose,
            logdir    = logdir
        )


# from .listnet import ListNet
