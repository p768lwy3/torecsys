from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import FMLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel


class FactorizationMachineSupportedNeuralNetworkModel(CtrBaseModel):
    r"""
    Model class of Factorization Machine supported Neural Network (FMNN).
    
    Factorization Machine supported Neural Network is a stack of Factorization Machine and Dense  Network, with the
    following calculation:
    
    #. Calculate features interactions by factorization machine: 
    :math:`y_{FM} = \text{Sigmoid} ( w_{0} + \sum_{i=1}^{N} w_{i} x_{i} + \sum_{i=1}^{N} \sum_{j=i+1}^{N}
    <v_{i}, v_{j}> x_{i} x_{j} )`

    #. Feed interactions' representation to dense network: 
    :math:`y_{i} = \text{Activation} ( w_{i} y_{i - 1} + b_{i} )`, where 
    :math:`y_{0} = y_{FM}` for the inputs of the first layer in dense network.

    :Reference:

    #. `Weinan Zhang et al, 2016. Deep Learning over Multi-field Categorical Data: A Case Study on User Response
        Prediction <https://arxiv.org/abs/1601.02376>`_.
    
    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 deep_output_size: int,
                 deep_layer_sizes: List[int],
                 fm_dropout_p: Optional[float] = 0.0,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize FactorizationMachineSupportedNeuralNetworkModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            deep_output_size (int): output size of dense network
            deep_layer_sizes (List[int]): layer sizes of dense network
            fm_dropout_p (float, optional): probability of Dropout in FM. Defaults to 0.0
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.fm = FMLayer(fm_dropout_p)

        cat_size = num_fields + embed_size
        self.deep = DNNLayer(
            inputs_size=cat_size,
            output_size=deep_output_size,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of FactorizationMachineSupportedNeuralNetworkModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), data_type = torch.float: linear Features tensors
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of FactorizationMachineSupportedNeuralNetworkModel
        """
        # Name the feat_inputs tensor for flatten
        feat_inputs.names = ('B', 'N', 'E',)

        # squeeze feat_inputs to shape = (B, N)
        if feat_inputs.dim() == 2:
            fm_first = feat_inputs
            fm_first.names = ('B', 'O',)
        elif feat_inputs.dim() == 3:
            # reshape feat_inputs from (B, N, 1) to (B, O = N)
            fm_first = feat_inputs.flatten(('N', 'E',), 'O')
        else:
            raise ValueError('Dimension of feat_inputs can only be 2 or 3')

        # pass to fm layer where its returns' shape = (B, O = E)
        fm_second = self.fm(emb_inputs)

        # concat into a tensor with shape = (B, O = N + E)
        fm_out = torch.cat([fm_first, fm_second], dim='O')

        # feed-forward to deep neural network, return shape = (B, O)
        outputs = self.deep(fm_out)

        # since autograd does not support Named Tensor at this stage, drop the name of output tensor.
        outputs = outputs.rename(None)

        return outputs
