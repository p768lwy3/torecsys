import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor

class FieldAwareFactorizationMachineLayer(nn.Module):
    """Layer class of Field-aware Factorication Machine (FFM).
    
    Field-aware Factorication Machine is purposed by Yuchin Juan et al, 2016, to calculate 
    element-wise cross feature interaction per field of sparse fields by using dot product 
    between field-wise feature tensors.

    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.
    
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 num_fields : int,
                 dropout_p  : float = 0.0):
        r"""Initialize FieldAwareFactorizationMachineLayer

        Args:
            num_fields (int): Number of inputs' fields
            dropout_p (float, optional): Probability of Dropout in FFM. 
                Defaults to 0.0.
        
        Attributes:
            num_fields (int): Number of inputs' fields.
            dropout (torch.nn.Module): Dropout layer.
        """
        # Refer to parent class
        super(FieldAwareFactorizationMachineLayer, self).__init__()

        # Bind num_fields to num_fields
        self.num_fields = num_fields

        # Initialize dropout
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of FieldAwareFactorizationMachineLayer

        Args:
            field_emb_inputs (T), shape = (B, N * N, E), dtype = torch.float: Field aware embedded features tensors.
        
        Returns:
            T, shape = (B, NC2, E), dtype = torch.float: Output of FieldAwareFactorizationMachineLayer
        """
        # Initialize list to store tensors temporarily for output 
        outputs = list()

        # Chunk field_emb_inputs into num_fields parts
        # inputs: field_emb_inputs, shape = (B, N * N , E)
        # output: field_emb_inputs, shape = (B, Nx = N, Ny = N, E)
        field_emb_inputs = field_emb_inputs.unflatten("N", [("Nx", self.num_fields), ("Ny", self.num_fields)])
        field_emb_inputs.names = None

        # Calculate dot-product between e_{i, fj} and e_{j, fi}
        # inputs: field_emb_inputs, shape = (B, Nx = N, Ny = N, E)
        # output: output, shape = (B, N = 1, E)
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ## output = field_emb_inputs[:, j, i] * field_emb_inputs[:, i, j]
                fij = field_emb_inputs[:, i, j]
                fji = field_emb_inputs[:, j, i]
                output = torch.einsum("ij,ij->ij", [fij, fji])
                output.names = ("B", "E")
                output = output.unflatten("B", [("B", output.size("B")), ("N", 1)])
                outputs.append(output)
        
        # Concat outputs into a tensor
        # inputs: output, shape = (B, N = 1, E)
        # output: outputs, shape = (B, NC2, E)
        outputs = torch.cat(outputs, dim="N")

        # Apply dropout
        # inputs: outputs, shape = (B, NC2, E)
        # output: outputs, shape = (B, NC2, E)
        outputs = self.dropout(outputs)
        
        return outputs
