from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import FMLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel


class NeuralFactorizationMachineModel(CtrBaseModel):
    """
    Model class of Neural Factorization Machine (NFM).
    
    Neural Factorization Machine is a model to pool embedding tensors from (B, N, E) to (B, 1, E) with Factorization
    Machine (FM) as an embedder of Deep Neural Network, i.e. a stack of factorization machine and dense network.

    :Reference:

    #. `Xiangnan He et al, 2017. Neural Factorization Machines for Sparse Predictive Analytics
        <https://arxiv.org/abs/1708.05027>`_.

    """

    def __init__(self,
                 embed_size: int,
                 deep_layer_sizes: List[int],
                 use_bias: Optional[bool] = True,
                 fm_dropout_p: Optional[float] = None,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize NeuralFactorizationMachineModel.
        
        Args:
            embed_size (int): size of embedding tensor
            deep_layer_sizes (List[int]): layer sizes of dense network
            use_bias (bool, optional): whether the bias constant is concatenated to the input. Defaults to True
            fm_dropout_p (float, optional): probability of Dropout in FM. Defaults to None
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (Callable[[T], T], optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.sequential = nn.Sequential()
        self.sequential.add_module('B_interaction', FMLayer(fm_dropout_p))
        self.sequential.add_module('Deep', DNNLayer(
            output_size=1,
            layer_sizes=deep_layer_sizes,
            inputs_size=embed_size,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        ))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1, 1), names=('B', 'O',)))
            nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of NeuralFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), data_type = torch.float: features tensors
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, 1), data_type = torch.float: output of NeuralFactorizationMachineModel
        """
        # Name the emb_inputs tensor for flatten
        feat_inputs.names = ('B', 'N', 'E',)

        # Aggregate feat_inputs on dimension N and rename dimension E to O
        # inputs: feat_inputs, shape = (B, N, E)
        # output: nfm_first, shape = (B, O = 1)
        nfm_first = feat_inputs.sum(dim='N').rename(E='O')

        # Calculate with sequential model forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: nfm_second, shape = (B, O = 1)
        nfm_second = self.sequential(emb_inputs)

        # Add up nfm_second, nfm_first and bias
        # inputs: nfm_second, shape = (B, O = 1)
        # inputs: nfm_first, shape = (B, O = 1)
        # inputs: bias, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = nfm_second + nfm_first
        if self.use_bias:
            outputs += self.bias

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
