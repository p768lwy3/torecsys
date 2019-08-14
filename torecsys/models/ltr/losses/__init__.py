r"""torecsys.models.ltr.losses is a sub module of loss functions used in Learning to Rank model
"""

from torecsys.losses import _Loss

class _RankingLoss(_Loss):
    def __init__(self):
        super(_RankingLoss, self).__init__()

from .functional import *

# from .groupwise_ranking_loss import *
from .pointwise_ranking_loss import *
from .pairwise_ranking_loss import *
