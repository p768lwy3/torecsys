from . import _CtrModel
from torecsys.layers import AttentionalFactorizationMachineLayer
from torecsys.utils.decorator import jit_experimental
from functools import partial
import torch
import torch.nn as nn
from typing import Dict

class AttentionalFactorizationMachineModel(_CtrModel):
    r"""AttentionalFactorizationMachineModel is a model of attentional factorization machine,
    which calculate prediction by summing up bias, linear terms and attentional factorization 
    machine values.

    :Reference:

    #. `Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks∗ <https://arxiv.org/abs/1708.04617>`_.

    """
    def __init__(self,
                 embed_size : int,
                 num_fields : int,
                 attn_size  : int,
                 dropout_p  : float = 0.0):
        r"""initialize Attention Factorization Machine Model
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in input
            attn_size (int): attention layer size
            dropout_p (float, optional): dropout probability after AFM layer. Defaults to 0.0.
        """
        super(AttentionalFactorizationMachineModel, self).__init__()
        
        # initialize attentional factorization machine layer
        self.afm = AttentionalFactorizationMachineLayer(embed_size, num_fields, attn_size, dropout_p)

        # initialize bias parameter
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.bias.data)
        
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of AttentionalFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: linear terms of fields, which can be get from nn.Embedding(embed_size=1)
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: second order terms of fields that will be passed into afm layer and can be get from nn.Embedding(embed_size=E)
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: predicted values of afm model
        """
        # first_order's shape = (B, N, 1)
        # output's shape = (B, 1)
        linear_out = feat_inputs.sum(dim=1)

        # second_order's shape = (B, N, E)
        # output's shape = (B, 1, E)
        # aggregate afm_out with dim = 2, and the output's shape = (B, 1)
        afm_out = self.afm(emb_inputs).sum(dim=2)

        # sum up bias, linear_out and afm_out to output
        outputs = self.bias + linear_out + afm_out

        return outputs
    