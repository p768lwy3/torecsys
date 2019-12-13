r"""torecsys.models.ltr is a sub module of the implementations of the whole models of Learning-to-Rank algorithm
"""

from .. import _Model

class _LtrModel(_Model):
    def __init__(self):
        super(_LtrModel, self).__init__()

class _RerankingModel(_LtrModel):
    def __init__(self):
        super(_RerankingModel, self).__init__()

from .personalized_reranking import PersonalizedRerankingModel
