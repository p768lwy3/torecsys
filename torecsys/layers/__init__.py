"""
torecsys.layers is a sub model of implementation of layers in recommendation system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch.nn as nn


class BaseLayer(nn.Module, ABC):
    """
    Base Layer for the torecsys module
    """

    def __init__(self, **kwargs):
        """
        Initializer for BaseLayer

        Args:
            **kwargs: kwargs
        """
        super(BaseLayer, self).__init__()

    @property
    @abstractmethod
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        """
        Get inputs size of the layer

        Returns:
            Dict[str, Tuple[str, ...]]: dictionary of inputs_size
        """
        raise NotImplemented('not implemented')

    @property
    @abstractmethod
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        """
        Get outputs size of the layer

        Returns:
            Dict[str, Tuple[str, ...]]: dictionary of outputs_size
        """
        raise NotImplemented('not implemented')


__all__ = [
    'AFMLayer',
    'AttentionalFactorizationMachineLayer',
    'BaseLayer',
    'BiasEncodingLayer',
    'BilinearNetworkLayer',
    'BilinearInteractionLayer',
    'CENLayer',
    'CINLayer',
    'ComposeExcitationNetworkLayer',
    'CompressInteractionNetworkLayer',
    'CrossNetworkLayer',
    'DenseLayer',
    'DNNLayer',
    'DynamicRoutingLayer',
    'FactorizationMachineLayer',
    'FeedForwardLayer',
    'FieldAwareFactorizationMachineLayer',
    'FFMLayer',
    'FMLayer',
    'FullyConnectLayer',
    'GeneralizedMatrixFactorizationLayer',
    'GMFLayer',
    'InnerProductNetworkLayer',
    'MOELayer',
    'MixtureOfExpertsLayer',
    'MultilayerPerceptionLayer',
    'OuterProductNetworkLayer',
    'PALLayer',
    'PositionEmbeddingLayer',
    'PositionBiasAwareLearningFrameworkLayer',
    'SENETLayer',
    'SqueezeAndExcitationNetworkLayer',
    'StarSpaceLayer',
    'WideLayer'
]

from torecsys.layers.ctr import *
from torecsys.layers.emb import *
from torecsys.layers.ltr import *
