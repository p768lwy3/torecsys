"""
torecsys.losses is a sub model of implementation of losses in recommendation system.
"""
from abc import ABC

import torch.nn as nn


class Loss(nn.Module, ABC):
    """
    General Loss class
    """

    def __init__(self):
        """
        Initialize Loss
        """
        super().__init__()


__all__ = [
    'AdaptiveHingeLoss',
    'BayesianPersonalizedRankingLoss',
    'HingeLoss',
    'ListnetLoss',
    'Loss',
    'PointwiseLogisticLoss',
    'TripletLoss',
    'SkipGramLoss'
]

from torecsys.losses.emb import *
from torecsys.losses.ltr import *
