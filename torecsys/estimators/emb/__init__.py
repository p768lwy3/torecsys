r"""torecsys.estimators.emb is a sub module of the estimators of embedding model
"""

from .. import _Estimator
from typing import Callable

class _EmbEstimator(_Estimator):
    r"""Base class of embedding estimator provide several functions would be called"""
    def __init__(self, **kwargs):
        super(_EmbEstimator, self).__init__(**kwargs)
