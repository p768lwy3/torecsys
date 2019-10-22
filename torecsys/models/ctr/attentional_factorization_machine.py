from . import _CtrModel
import torch
import torch.nn as nn
from torecsys.layers import AFMLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor


class AttentionalFactorizationMachineModel(_CtrModel):
    r"""Model class of Attentional Factorization Machine (AFM).
    
    :Reference:

    #. `Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks∗ <https://arxiv.org/abs/1708.04617>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size : int,
                 num_fields : int,
                 attn_size  : int,
                 dropout_p  : float = 0.0):
        r"""Initialize AttentionalFactorizationMachineModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            attn_size (int): Size of attention layer
            dropout_p (float, optional): Probability of Dropout in AFM. 
                Defaults to 0.0.
        
        Attributes:
            afm (nn.Module): Module of AFM layer.
            bias (nn.Parameter): Parameter of bias in output projection.
        """
        # refer to parent class
        super(AttentionalFactorizationMachineModel, self).__init__()
        
        # initialize attentional factorization machine layer
        self.afm = AFMLayer(embed_size, num_fields, attn_size, dropout_p)

        # initialize bias parameter
        self.bias = nn.Parameter(torch.zeros(size=(1, 1), names=("B", "O")))
        nn.init.uniform_(self.bias.data)
        
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of AttentionalFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: Linear Features tensors.
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of AttentionalFactorizationMachineModel.
        """
        # feat_inputs's shape = (B, N, 1) -> linear_out's shape = (B, O = 1)
        linear_out = feat_inputs.sum(dim="N")
        linear_out.names = ("B", "O")

        # emb_inputs's shape = (B, N, E) -> afm_out's shape = (B, E)
        # then aggregate afm_out on dimension = "E" -> output's shape = (B, 1)
        afm_out, _ = self.afm(emb_inputs)
        afm_out = afm_out.sum(dim="E", keepdim=True)
        afm_out.names = ("B", "O")

        # sum up bias, linear_out and afm_out to output
        outputs = self.bias + linear_out + afm_out

        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        outputs = outputs.rename(None)

        return outputs
    