from . import _CtrModule
from ..layers import AttentionalFactorizationMachineLayer
from torecsys.utils.logging.decorator import jit_experimental
from functools import partial
import torch
import torch.nn as nn
from typing import Dict

class AttentionalFactorizationMachineModule(_CtrModule):

    def __init__(self,
                 embed_size    : int,
                 num_fields    : int,
                 attn_size     : int,
                 dropout_p     : float = 0.1,
                 bias          : bool = True,
                 output_method : str = "concatenate",
                 output_size   : int = 1):

        super(AttentionalFactorizationMachineModule, self).__init__()

        if bias:
            # output size = fm output size + linear output size + bias
            self.bias = nn.Parameter(torch.zeros(1))
            nn.init.uniform_(self.bias.data)
            cat_size = embed_size + num_fields + 1
        else:
            # output size = fm output size + linear output size
            self.bias = None
            cat_size = embed_size + num_fields
        
        self.afm = AttentionalFactorizationMachineLayer(embed_size, num_fields, attn_size, dropout_p)

        if output_method == "concatenate":
            self.fc = nn.Linear(cat_size, output_size)
        elif output_method == "sum":
            self.fc = partial(torch.sum, dim=1, keepdim=True)
        else:
            raise ValueError("output_method %s is not allowed." % output_method)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:

        # get batch size
        batch_size = inputs["first_order"].size(0)

        # first_order's shape = (batch size, number of fields, 1)
        # which could be get by nn.Embedding(vocab size, 1)
        linear_out = inputs["first_order"]

        # second_order's shape = (batch size, number of fields, embed size)
        # which could be get by nn.Embedding(vocab size, embed size)
        afm_out = self.afm(inputs["second_order"])

        # reshape afm_out to (batch size, embed size)
        afm_out = afm_out.view(batch_size, -1)

        # reshape linear_out to (batch size, number of fields)
        linear_out = linear_out.view(batch_size, -1)

        # repeat and reshape bias to shape = (batch size, 1)
        if self.bias is not None:
            # shape = (batch size, number of fields * embed size + number of fields + 1)
            bias = self.bias.repeat(batch_size).view(batch_size, 1)
            outputs = torch.cat([afm_out, linear_out, bias], dim=1)
        else:
            # shape = (batch size, number of fields * embed size + number of fields)
            outputs = torch.cat([afm_out, linear_out], dim=1)
            
        # fully-connected dense layer for output, return (batch size, output size)
        # or sum with second dimension, return (batch size, 1)
        outputs = self.fc(outputs)

        return outputs
    