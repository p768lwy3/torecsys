from typing import Optional

import torch
import torch.nn as nn

from torecsys.layers import FFMLayer
from torecsys.models.ctr import CtrBaseModel


class FieldAwareFactorizationMachineModel(CtrBaseModel):
    """Model class of Field-aware Factorization Machine (FFM).
    
    Field-aware Factorization Machine is a model to calculate the interaction of features for each field with different
    embedding vectors, instead of a universal vectors.

    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction
        <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.

    """

    def __init__(self,
                 num_fields: int,
                 dropout_p: Optional[float] = 0.0):
        """
        Initialize FieldAwareFactorizationMachineModel
        
        Args:
            num_fields (int): number of inputs' fields
            dropout_p (float, optional): probability of Dropout in FFM. Defaults to 0.0
        """
        super().__init__()

        self.ffm = FFMLayer(num_fields, dropout_p=dropout_p)
        self.bias = nn.Parameter(torch.zeros((1, 1,), names=('B', 'O',)))
        nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of FieldAwareFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), data_type = torch.float: linear Features tensors
            field_emb_inputs (T), shape = (B, N * N, E), data_type = torch.float: field aware embedded features tensors
        
        Returns:
            torch.Tensor, shape = (B, O), data_type = torch.float: output of FieldAwareFactorizationMachineModel
        """
        # Name the feat_inputs tensor for flatten
        feat_inputs.names = ('B', 'N', 'E',)

        # Get batch size from field_emb_inputs
        b = feat_inputs.size('B')

        # Aggregate feat_inputs on dimension N and rename dimension E to O
        # Hence, ffm_first's shape = (B, O = 1)
        # inputs: feat_inputs, shape = (B, N, E = 1)
        # output: ffm_first, shape = (B, O = 1)
        ffm_first = feat_inputs.sum(dim='N').rename(E='O')

        # inputs: field_emb_inputs, shape = (B, N * N, E)
        # output: ffm_second, shape = (B, N, E)
        ffm_second = self.ffm(field_emb_inputs)

        # Aggregate ffm_second on dimension [N, E], then reshape the sum from (B, ) to (B, O = 1)
        # inputs: ffm_second, shape = (B, N, E)
        # output: ffm_second, shape = (B, O = 1)
        ffm_second = ffm_second.sum(dim=('N', 'E',)).unflatten('B', (('B', b,), ('O', 1,),))

        # Add up bias, fm_first and ffm_second
        # inputs: ffm_second, shape = (B, O = 1)
        # inputs: ffm_first, shape = (B, O = 1)
        # inputs: bias, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = ffm_second + ffm_first + self.bias

        # since autograd does not support Named Tensor at this stage, drop the name of output tensor.
        outputs = outputs.rename(None)

        return outputs
