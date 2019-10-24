import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor

class FactorizationMachineLayer(nn.Module):
    """Layer class of Factorization Machine (FM). 
    
    Factorization Machine is purposed by :title:`Steffen Rendle, 2010`[1] to calculate low 
    dimension cross features interactions of sparse field by using a general form of matrix 
    factorization.

    :Reference:

    #. `Steffen Rendle, 2010. Factorization Machine <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_.
    
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 dropout_p: float = 0.0):
        r"""Initialize FactorizationMachineLayer
        
        Args:
            dropout_p (float, optional): Probability of Dropout in FM. 
                Defaults to 0.0.
        
        Arguments:
            dropout (torch.nn.Module): Dropout layer.
        """
        # Refer to parent class
        super(FactorizationMachineLayer, self).__init__()

        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of FactorizationMachineLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of FactorizationMachineLayer
        """
        # Square summed embedding
        # inputs: emb_inputs, shape = (B, N, E)
        # output: squared_sum_embs, shape = (B, E)
        squared_sum_embs = (emb_inputs.sum(dim="N")) ** 2
        
        # Sum squared embedding
        # inputs: emb_inputs, shape = (B, N, E)
        # output: sum_squared_embs, shape = (B, E)
        sum_squared_embs = (emb_inputs ** 2).sum(dim="N")
        
        # Calculate outputs of fm
        # inputs: squared_sum_embs, shape = (B, E)
        # inputs: sum_squared_embs, shape = (B, E)
        # output: outputs, shape = (B, E)
        outputs = 0.5 * (squared_sum_embs - sum_squared_embs)

        # Apply dropout
        # inputs: outputs, shape = (B, E)
        # output: outputs, shape = (B, E)
        outputs = self.dropout(outputs)        
        outputs.names = ("B", "O")
        
        return outputs
        