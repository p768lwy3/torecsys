from . import _CtrModel
import torch
import torch.nn as nn
from torecsys.layers import AFMLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor


class AttentionalFactorizationMachineModel(_CtrModel):
    r"""AttentionalFactorizationMachineModel is a model of attentional factorization machine,
    which calculate prediction by summing up bias, linear terms and attentional factorization 
    machine values.

    :Reference:

    #. `Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks∗ <https://arxiv.org/abs/1708.04617>`_.

    """
    @no_jit_experimental_by_namedtensor
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
        
        Attributes:
            afm (nn.Module): module of AFM layer
            bias (nn.Parameter): parameter of bias in output projection
        """
        super(AttentionalFactorizationMachineModel, self).__init__()
        
        # initialize attentional factorization machine layer
        self.afm = AFMLayer(embed_size, num_fields, attn_size, dropout_p)

        # initialize bias parameter
        self.bias = nn.Parameter(torch.zeros(size=(1, 1), names=("B", "O")))
        nn.init.uniform_(self.bias.data)
        
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of AttentionalFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, 1, N), dtype = torch.float: linear terms of fields, which can be get from nn.Embedding(embed_size=1)
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: second order terms of fields that will be passed into afm layer and can be get from nn.Embedding(embed_size=E)
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: predicted values of afm model
        """
        # first_order's shape = (B, N) -> output's shape = (B, O = 1)
        linear_out = feat_inputs.sum(dim="N")
        linear_out.names = ("B", "O")

        # second_order's shape = (B, N, E) -> afm_out's shape = (B, E)
        # aggregate afm_out by dim = "E" -> output's shape = (B, 1)
        afm_out, _ = self.afm(emb_inputs)
        afm_out = afm_out.sum(dim="E", keepdim=True)
        afm_out.names = ("B", "O")

        # sum up bias, linear_out and afm_out to output
        outputs = self.bias + linear_out + afm_out

        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        outputs = outputs.rename(None)

        return outputs
    