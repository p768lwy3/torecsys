r"""torecsys.estimators.emb is a sub module of the estimators of embedding model
"""

from .. import _Estimator

class _EmbEstimator(_Estimator):
    r"""Base class of embedding estimator provide several functions would be called"""
    def __init__(self, 
                 model    : Callable,
                 loss     : Callable,
                 optimizer: Callable, 
                 epochs   : int,
                 verbose  : int,
                 logdir   : logdir = None):
        
        super(_EmbEstimator, self).__init__(
            model     = model,
            loss      = loss,
            optimizer = optimizer,
            epochs    = epochs,
            verbose   = verbose,
            logdir    = logdir
        )
