from . import _CtrModel
from torecsys.layers import FMLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn


class FactorizationMachineModel(_CtrModel):
    r"""Model class of Factorization Machine (FM).
    
    Factoization Machine is a model to calculate interactions between fields in the following way: 
    :math:`\^{y}(x) := b_{0} + \sum_{i=1}^{n} w_{i} x_{i} + \sum_{i=1}^{n} \sum_{j=1+1}^{n} <v_{i},v_{j}> x_{i} x_{j}`_.

    :Reference:

    #. `Steffen Rendle, 2010. Factorization Machine <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 embed_size    : int,
                 num_fields    : int,
                 dropout_p     : float = 0.0):
        r"""Initialize FactorizationMachineModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            dropout_p (float, optional): Probability of Dropout in FM. 
                Defaults to 0.0.
        
        Attributions:
            fm (nn.Module): Module of factorization machine layer.
            bias (nn.Parameter): Parameter of bias in output projection.
        """
        # refer to parent class
        super(FactorizationMachineModel, self).__init__()

        # initialize fm layer
        self.fm = FMLayer(dropout_p)
        
        # initialize bias parameter
        self.bias = nn.Parameter(torch.zeros((1, 1), names=("B", "O")))
        nn.init.uniform_(self.bias.data)
    
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of FactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: Linear Features tensors.
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            torch.Tensor, shape = (B, O), dtype = torch.float: Output of FactorizationMachineModel.
        """
        # feat_inputs'shape = (B, N, 1) and reshape to (B, N)
        ## fm_first = feat_inputs.sum(dim=1)
        fm_first = feat_inputs.sum(dim="N").rename(E="O")

        # pass to fm layer where its returns' shape = (B, E)
        fm_second = self.fm(emb_inputs).sum(dim="O", keepdim=True)
            
        # sum bias, fm_first, fm_second and get fm outputs with shape = (B, 1)
        outputs = fm_second + fm_first + self.bias

        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        outputs = outputs.rename(None)

        return outputs
