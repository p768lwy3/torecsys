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
        # Refer to parent class
        super(AttentionalFactorizationMachineModel, self).__init__()
        
        # Initialize attentional factorization machine layer
        self.afm = AFMLayer(embed_size, num_fields, attn_size, dropout_p)

        # Initialize bias parameter
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
        # Aggregate feat_inputs on dimension N
        # inputs: feat_inputs, shape = (B, N, 1)
        # output: linear_out, shape = (B, O = 1)
        afm_first = feat_inputs.sum(dim="N")
        afm_first.names = ("B", "O")

        # Calculate with AFM layer forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: afm_second, shape = (B, E)
        afm_second, _ = self.afm(emb_inputs)

        # Aggregate afm_second on dimension E
        # inputs: afm_second, shape = (B, E)
        # output: afm_second, shape = (B, O = 1)
        afm_second = afm_second.sum(dim="E", keepdim=True).rename(E="O")

        # Add up afm_second, afm_first and bias
        # inputs: afm_second, shape = (B, O = 1)
        # inputs: afm_first, shape = (B, O = 1)
        # inputs: bias, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = afm_second + afm_first + self.bias

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
    