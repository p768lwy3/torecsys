import torch
import torch.nn as nn

from torecsys.utils.decorator import no_jit_experimental_by_namedtensor


class FactorizationMachineLayer(nn.Module):
    r"""Layer class of Factorization Machine (FM). 
    
    Factorization Machine is purposed by Steffen Rendle, 2010 to calculate low dimension cross 
    features interactions of sparse field by using a general form of matrix factorization.

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
        
        Attributes:
            dropout (torch.nn.Module): Dropout layer.
        """
        # refer to parent class
        super(FactorizationMachineLayer, self).__init__()

        # initialize dropout layer
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
        # output: squared_sum_emb, shape = (B, E)
        squared_sum_emb = (emb_inputs.sum(dim="N")) ** 2

        # Sum squared embedding
        # inputs: emb_inputs, shape = (B, N, E)
        # output: sum_squared_emb, shape = (B, E)
        sum_squared_emb = (emb_inputs ** 2).sum(dim="N")

        # Calculate outputs of fm
        # inputs: squared_sum_emb, shape = (B, E)
        # inputs: sum_squared_emb, shape = (B, E)
        # output: outputs, shape = (B, E)
        outputs = 0.5 * (squared_sum_emb - sum_squared_emb)

        # Apply dropout
        # inputs: outputs, shape = (B, E)
        # output: outputs, shape = (B, E)
        outputs = self.dropout(outputs)
        outputs.names = ("B", "O")

        return outputs
