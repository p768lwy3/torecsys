from . import _CtrModel
from torecsys.layers import FieldAwareFactorizationMachineLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


class FieldAwareFactorizationMachineModel(_CtrModel):
    r"""FieldAwareFactorizationMachineModel is a model of field-aware factorization machine (ffm), which is 
    to calculate the interaction of features for each field with different embedding vectors, instead of a 
    universal vectors. 

    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.

    """
    def __init__(self,
                 embed_size    : int,
                 num_fields    : int,
                 dropout_p     : float = 0.0):
        r"""initialize Field-Aware Factorization Machine Model
        
        Args:
            embed_size (int): embedding size
            num_fields (int): N in inputs
            dropout_p (float, optional): dropout probability of field-aware factorization machine layer. Defaults to 0.0.
        """
        super(FieldAwareFactorizationMachineModel, self).__init__()
            
        # initialize bias variable
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.bias.data)

        # initialize ffm layer
        self.ffm = FieldAwareFactorizationMachineLayer(dropout_p=dropout_p)

    
    def forward(self, feat_inputs: torch.Tensor, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of Field-Aware Factorization Machine Model 
        
        Args:
            feat_inputs (T), shape = (B, N, 1): first order outputs, i.e. outputs from nn.Embedding(V, 1)
            field_emb_inputs (T), shape = (B, N * N, E): field-aware second order outputs, :math:`x_{i, \text{field}_{j}}`
        
        Returns:
            torch.Tensor, shape = (B, O), dtype = torch.float: outputs of Deep Field-aware Factorization Machine Model
        """

        # feat_inputs'shape = (B, N, 1) and reshape to (B, N)
        ffm_first = feat_inputs.squeeze()

        # inputs' shape = (B, N * N, E) which could be get by ..inputs.base.FieldAwareIndexEmbedding
        # and output shape = (B, N, E)
        ffm_second = self.ffm(field_emb_inputs)

        # aggregate ffm_out in dimension [1, 2], where the shape = (B, 1)
        ffm_second = ffm_out.sum(dim=[1, 2])

        # sum bias, fm_first and ffm_sceond and getn fmm outputs with shape = (B, 1)
        outputs = ffm_second + ffm_first + self.bias

        return outputs
