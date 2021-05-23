from typing import Dict, Tuple

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class FieldAwareFactorizationMachineLayer(BaseLayer):
    """
    Layer class of Field-aware Factorization Machine (FFM).
    
    Field-aware Factorization Machine is purposed by Yuchin Juan et al, 2016, to calculate element-wise cross feature
    interaction per field of sparse fields by using dot product between field-wise feature tensors.

    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction
        <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.
    
    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'N^2', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'NC2', 'E',)
        }

    def __init__(self,
                 num_fields: int,
                 dropout_p: float = 0.0):
        """
        Initialize FieldAwareFactorizationMachineLayer

        Args:
            num_fields (int): number of inputs' fields
            dropout_p (float, optional): probability of Dropout in FFM. Defaults to 0.0
        """
        super().__init__()

        self.num_fields = num_fields
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of FieldAwareFactorizationMachineLayer

        Args:
            field_emb_inputs (T), shape = (B, N * N, E), data_type = torch.float: field aware embedded features tensors
        
        Returns:
            T, shape = (B, NC2, E), data_type = torch.float: output of FieldAwareFactorizationMachineLayer
        """
        # Name the inputs tensor for alignment
        field_emb_inputs.names = ('B', 'N', 'E',)

        # initialize list to store tensors temporarily for output
        outputs = []

        # chunk field_emb_inputs into num_fields parts
        # inputs: field_emb_inputs, shape = (B, N * N , E)
        # output: field_emb_inputs, shape = (B, Nx = N, Ny = N, E)
        field_emb_inputs = field_emb_inputs.unflatten('N', (('Nx', self.num_fields,), ('Ny', self.num_fields,),))
        field_emb_inputs.names = None

        # calculate dot-product between e_{i, fj} and e_{j, fi}
        # inputs: field_emb_inputs, shape = (B, Nx = N, Ny = N, E)
        # output: output, shape = (B, N = 1, E)
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                fij = field_emb_inputs[:, i, j]
                fji = field_emb_inputs[:, j, i]
                output = torch.einsum('ij,ij->ij', fij, fji)
                output.names = ('B', 'E',)
                output = output.unflatten('B', (('B', output.size('B'),), ('N', 1,),))
                outputs.append(output)

        # concat outputs into a tensor
        # inputs: output, shape = (B, N = 1, E)
        # output: outputs, shape = (B, NC2, E)
        outputs = torch.cat(outputs, dim='N')

        # apply dropout
        # inputs: outputs, shape = (B, NC2, E)
        # output: outputs, shape = (B, NC2, E)
        outputs = self.dropout(outputs)

        return outputs
