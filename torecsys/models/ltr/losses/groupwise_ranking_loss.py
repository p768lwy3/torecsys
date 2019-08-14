r"""torecsys.models.ltr.losses.groupwise_ranking_loss is a sub module of algorithms of 
groupwise ranking loss
"""

from . import _RankingLoss

class _GroupwiseRankingLoss(_RankingLoss):
    r"""Base Class of groupwise ranking loss
    """
    def __init__(self):
        super(_GroupwiseRankingLoss, self).__init__()
    
