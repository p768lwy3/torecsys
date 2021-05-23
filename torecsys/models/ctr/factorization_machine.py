from typing import Optional

import torch
import torch.nn as nn

from torecsys.layers import FMLayer
from torecsys.models.ctr import CtrBaseModel


class FactorizationMachineModel(CtrBaseModel):
    r"""
    Model class of Factorization Machine (FM).
    
    Factorization Machine is a model to calculate interactions between fields in the following way:
    :math:`\^{y}(x) := b_{0} + \sum_{i=1}^{n} w_{i} x_{i} + \sum_{i=1}^{n} \sum_{j=1+1}^{n} <v_{i},v_{j}> x_{i} x_{j}`

    :Reference:

    #. `Steffen Rendle, 2010. Factorization Machine <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_.

    """

    def __init__(self,
                 use_bias: bool = True,
                 dropout_p: Optional[float] = None):
        """
        Initialize FactorizationMachineModel
        
        Args:
            use_bias (bool, optional): whether the bias constant is added to the input. Defaults to True
            dropout_p (float, optional): probability of Dropout in FM. Defaults to None
        """
        super().__init__()

        self.fm = FMLayer(dropout_p)

        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros((1, 1,), names=('B', 'O',)))
            nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of FactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), data_type = torch.float: linear Features tensors
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            torch.Tensor, shape = (B, O), data_type = torch.float: output of FactorizationMachineModel
        """
        # Name the feat_inputs tensor for flatten
        feat_inputs.names = ('B', 'N', 'E',)

        # Aggregate feat_inputs on dimension N and rename dimension E to O
        # Hence, fm_first's shape = (B, O = 1)
        fm_first = feat_inputs.sum(dim='N').rename(E='O')

        # Pass to fm layer where its returns' shape = (B, E)
        fm_second = self.fm(emb_inputs).sum(dim='O', keepdim=True)

        # Sum bias, fm_first, fm_second and get fm outputs with shape = (B, 1)
        outputs = fm_second + fm_first
        if self.use_bias:
            outputs += self.bias

        # Since autograd does not support Named Tensor at this stage, drop the name of output tensor
        outputs = outputs.rename(None)

        return outputs
