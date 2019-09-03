from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


class FactorizationMachineLayer(nn.Module):
    r"""Layer class of Factorization Machine (FM) :title:`Steffen Rendle, 2010`[1], to calculate 
    low dimension cross features interactions for sparse field by using a general form of 
    matrix factorization.
    
    :Reference:
    
    #. `Steffen Rendle, 2010. Factorization Machine <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_.
    
    """
    @jit_experimental
    def __init__(self, 
                 dropout_p: float = 0.0):
        r"""Initialize FactorizationMachineLayer
        
        Args:
            dropout_p (float, optional): Probability of Dropout in FM. 
                Defaults to 0.0.
        
        Arguments:
            dropout (torch.nn.Module): Dropout layer.
        """
        # refer to parent class
        super(FactorizationMachineLayer, self).__init__()

        # initialize dropout layer before return
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of FactorizationMachineLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, 1, E), dtype = torch.float: Output of FactorizationMachineLayer
        """
        # squared sum embedding where output shape = (B, E)
        squared_sum_embs = (emb_inputs.sum(dim=1)) ** 2
        
        # sum squared embedding where output shape = (B, E)
        sum_squared_embs = (emb_inputs ** 2).sum(dim=1)
        
        # calculate output of fm
        outputs = 0.5 * (squared_sum_embs - sum_squared_embs)

        # apply dropout before return
        outputs = self.dropout(outputs)
        return outputs.unsqueeze(1)
        