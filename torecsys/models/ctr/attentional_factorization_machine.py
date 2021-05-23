from typing import Optional

import torch
import torch.nn as nn

from torecsys.layers import AFMLayer
from torecsys.models.ctr import CtrBaseModel


class AttentionalFactorizationMachineModel(CtrBaseModel):
    """
    Model class of Attentional Factorization Machine (AFM)
    
    :Reference:

    #. `Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via
    Attention Networks∗ <https://arxiv.org/abs/1708.04617>`_.

    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 attn_size: int,
                 use_bias: bool = True,
                 dropout_p: Optional[float] = None):
        """
        Initialize AttentionalFactorizationMachineModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            attn_size (int): size of attention layer
            use_bias (bool, optional): whether the bias constant is added to the input. Defaults to True
            dropout_p (float, optional): probability of Dropout in AFM. Defaults to None
        """
        super().__init__()

        self.afm = AFMLayer(embed_size, num_fields, attn_size, dropout_p)

        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(size=(1, 1,), names=('B', 'O',)))
            nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of AttentionalFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), data_type = torch.float: linear Features tensors.
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors.
        
        Returns:
            T, shape = (B, O), data_type = torch.float: Output of AttentionalFactorizationMachineModel.
        """
        # Name the feat_inputs tensor for flatten
        feat_inputs.names = ('B', 'N', 'E',)

        # Aggregate feat_inputs on dimension N
        # inputs: feat_inputs, shape = (B, N, 1)
        # output: linear_out, shape = (B, O = 1)
        afm_first = feat_inputs.sum(dim='N').rename(E='O')

        # Calculate with AFM layer forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: afm_second, shape = (B, E)
        afm_second, _ = self.afm(emb_inputs)

        # Aggregate afm_second on dimension E
        # inputs: afm_second, shape = (B, E)
        # output: afm_second, shape = (B, O = 1)
        afm_second = afm_second.sum(dim='E', keepdim=True).rename(E='O')

        # Add up afm_second, afm_first and bias
        # inputs: afm_second, shape = (B, O = 1)
        # inputs: afm_first, shape = (B, O = 1)
        # inputs: bias, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = afm_second + afm_first
        if self.use_bias:
            outputs += self.bias

        # Drop names of outputs, since auto grad doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
