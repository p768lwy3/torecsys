r"""torecsys.estimators.ctr. is a sub module of the estimators of click-through-rate prediction model
"""

from .. import _Estimator 

class _CtrEstimator(_Estimator):
    r"""Base class of click through rate prediction estimator provide several functions would 
    be called"""
    def __init__(self, 
                 model    : Callable,
                 loss     : Callable,
                 optimizer: Callable, 
                 epochs   : int,
                 verbose  : int,
                 logdir   : logdir = None):
        
        super(_CtrEstimator, self).__init__(
            model     = model,
            loss      = loss,
            optimizer = optimizer,
            epochs    = epochs,
            verbose   = verbose,
            logdir    = logdir
        )
