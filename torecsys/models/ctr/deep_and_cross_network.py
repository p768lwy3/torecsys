from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import CrossNetworkLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel


class DeepAndCrossNetworkModel(CtrBaseModel):
    """
    Model class of Deep & Cross Network (DCN), which is a concatenation of dense network (deep part) and
    cross network (cross part).

    :Reference:

    #. `Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`_.
    
    """

    def __init__(self,
                 inputs_size: int,
                 num_fields: int,
                 deep_output_size: int,
                 deep_layer_sizes: List[int],
                 cross_num_layers: int,
                 output_size: int = 1,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize DeepAndCrossNetworkModel
        
        Args:
            inputs_size (int): inputs size of dense network and cross network, i.e. number of fields * embedding size
            deep_output_size (int): output size of dense network
            deep_layer_sizes (List[int]): layer sizes of dense network
            cross_num_layers (int): number of layers of Cross Network
            output_size (int, optional): output size of model, i.e. output size of the projection layer. Defaults to 1
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.deep = DNNLayer(
            inputs_size=inputs_size,
            output_size=deep_output_size,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )
        self.cross = CrossNetworkLayer(
            inputs_size=inputs_size,
            num_layers=cross_num_layers
        )
        cat_size = (deep_output_size + inputs_size) * num_fields
        self.fc = nn.Linear(cat_size, output_size)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of DeepAndCrossNetworkModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of DeepAndCrossNetworkModel
        """
        # Forward calculate with cross layer
        # inputs: emb_inputs, shape = (B, N, E)
        # output: cross_out, shape = (B, N, E)
        cross_out = self.cross(emb_inputs)

        # Forward calculate with dense layer
        # inputs: emb_inputs, shape = (B, N * E)
        # output: deep_out, shape = (B, N, O = Od)
        deep_out = self.deep(emb_inputs)

        # Concatenate on dimension = O
        # inputs: cross_out, shape = (B, N, E)
        # inputs: deep_out, shape = (B, N, Od)
        # output: outputs, shape = (B, N, O = E + Od)
        outputs = torch.cat([cross_out, deep_out], dim='O')

        # Flatten emb_inputs
        # inputs: emb_inputs, shape = (B, N, O)
        # output: emb_inputs, shape = (B, N * O)
        outputs = outputs.flatten(('N', 'O',), 'O')

        # Calculate with linear layer forwardly
        # inputs: outputs, shape = (B, N * O)
        # output: outputs, shape = (B, Ofc)
        outputs = self.fc(outputs)
        outputs.names = ('B', 'O',)

        # Drop names of outputs, since auto grad doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
