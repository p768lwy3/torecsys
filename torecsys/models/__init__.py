"""
torecsys.models is a sub model of implementation with a whole models in recommendation system
"""
from abc import ABC

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()


__all__ = [
    'AttentionalFactorizationMachineModel',
    'BaseModel',
    'DeepAndCrossNetworkModel',
    'DeepFactorizationMachineModel',
    'DeepFieldAwareFactorizationMachineModel',
    'DeepMatchingCorrelationPredictionModel',
    'DeepMixtureOfExpertsModel',
    'ElaboratedEntireSpaceSupervisedMultiTaskModel',
    'EntireSpaceMultiTaskModel',
    'FactorizationMachineModel',
    'FactorizationMachineSupportedNeuralNetworkModel',
    "FeatureImportanceAndBilinearFeatureInteractionNetwork",
    'FieldAttentiveDeepFieldAwareFactorizationMachineModel',
    'FieldAwareFactorizationMachineModel',
    'LogisticRegressionModel',
    'MatrixFactorizationModel',
    'MultiGateMixtureOfExpertsModel',
    'NeuralCollaborativeFilteringModel',
    'NeuralFactorizationMachineModel',
    'PersonalizedReRankingModel',
    'PositionBiasAwareLearningFrameworkModel',
    'ProductNeuralNetworkModel',
    'StarSpaceModel',
    'WideAndDeepModel',
    'XDeepFactorizationMachineModel'
]

from torecsys.models.ctr import *
from torecsys.models.emb import *
from torecsys.models.ltr import *
