"""
torecsys.losses.ltr is a sub model of implementation of losses in learning-to-rank.
"""

from torecsys.losses import Loss


class RankingLoss(Loss):
    def __init__(self):
        super().__init__()


__all__ = [
    'AdaptiveHingeLoss',
    'BayesianPersonalizedRankingLoss',
    'HingeLoss',
    'ListnetLoss',
    'PointwiseLogisticLoss',
    'RankingLoss',
    'TripletLoss'
]

from torecsys.losses.ltr.groupwise_ranking_loss import *
from torecsys.losses.ltr.pointwise_ranking_loss import *
from torecsys.losses.ltr.pairwise_ranking_loss import *
