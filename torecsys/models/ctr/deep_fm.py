from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import FMLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel


class DeepFactorizationMachineModel(CtrBaseModel):
    """
    Model class of Deep Factorization Machine (Deep FM).
    
    Deep Factorization Machine is a model proposed by Huawei in 2017, which sum up outputs of factorization machine
    and fully-connected dense network directly, to gain the advantages from two different models of two different
    objectives, i.e. to gain the explainable power in high dimension of dense network, and to gain the explainable
    power in low dimension of factorization machine.

    :Reference:

    #. `Huifeng Guo et al, 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    <https://arxiv.org/abs/1703.04247>`_.
    
    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 deep_layer_sizes: List[int],
                 fm_dropout_p: Optional[float] = None,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize DeepFactorizationMachineModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            deep_layer_sizes (List[int]): layer sizes of dense network
            fm_dropout_p (float, optional): probability of Dropout in FM. Defaults to 0.0
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.fm = FMLayer(fm_dropout_p)
        self.deep = DNNLayer(
            inputs_size=num_fields * embed_size,
            output_size=1,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of DeepFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), data_type = torch.float: linear Features tensors
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of DeepFactorizationMachineModel
        """
        # Name feat_inputs and emb_inputs tensors for flatten
        feat_inputs.names = ('B', 'N', 'E',)
        emb_inputs.names = ('B', 'N', 'E',)

        # Reshape feat_inputs 
        # inputs: feat_inputs, shape = (B, N, 1)
        # output: fm_first, shape = (B, O = N)
        fm_first = feat_inputs.flatten(('N', 'E',), 'O')

        # Calculate with fm layer forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: fm_second, shape = (B, O = E)
        fm_second = self.fm(emb_inputs)

        # Concatenate fm_second and fm_first on dimension O
        # inputs: fm_second, shape = (B, O = E)
        # inputs: fm_first, shape = (B, O = N)
        # output: fm_out, shape = (B, O = E + N)
        fm_out = torch.cat([fm_second, fm_first], dim='O')

        # Aggregate fm_out on dimension O
        # inputs: fm_out, shape = (B, O)
        # output: fm_out, shape = (B, O = 1)
        fm_out = fm_out.sum(dim='O', keepdim=True)

        # Flatten dffm_second
        # inputs: emb_inputs, shape = (B, N, E)
        # output: deep_in, shape = (B, N * E)
        deep_in = emb_inputs.flatten(('N', 'E',), 'E')

        # Forward Calculate with dense layer
        # inputs: emb_inputs, shape = (B, N * E)
        # output: deep_out, shape = (B, O = 1)
        deep_out = self.deep(deep_in)

        # Add up deep_out and fm_out
        # inputs: deep_out, shape = (B, O = 1)
        # inputs: fm_out, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = deep_out + fm_out

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
