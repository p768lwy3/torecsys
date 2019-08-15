from . import _Module
from ..layers import FieldAwareFactorizationMachineLayer
from functools import partial
import torch
import torch.nn as nn
from typing import Dict

class FieldAwareFactorizationMachineModule(_Module):
    r""""""
    def __init__(self,
                 embed_size    : int,
                 num_fields    : int,
                 dropout_p     : float = 0.1,
                 output_method : str = "concatenate",
                 output_size   : int = 1):
        r"""[summary]
        
        Args:
            embed_size (int): [description]
            num_fields (int): [description]
            dropout_p (float, optional): [description]. Defaults to 0.1.
            output_method (str, optional): [description]. Defaults to "concatenate".
            output_size (int, optional): [description]. Defaults to 1.
        """
        super(FieldAwareFactorizationMachineModule, self).__init__()
        if bias:
            # output size = fm output size + linear output size + bias
            self.bias = nn.Parameter(torch.zeros(1))
            nn.init.xavier_uniform_(self.bias.data)
            output_size = embed_size + num_fields + 1
        else:
            # output size = fm output size + linear output size
            self.bias = None
            output_size = embed_size + num_fields
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""[summary]
        
        Args:
            inputs (Dict[str, torch.Tensor]): [description]
        
        Returns:
            torch.Tensor: [description]
        """
        # get batch size
        batch_size = inputs["first_order"].size(0)

        # inputs' shape = (batch size, number of fields, 1)
        # which could be get by nn.Embedding(vocab size, 1)
        linear_out = inputs["first_order"]

        # inputs' shape = (batch size, number of fields * number of fields, embed size)
        # which could be get by trs.models.inputs.FieldAwareIndexEmbedding
        ffm_out = self.ffm(inputs["second_order"])

        # reshape ffm_out to (batch size, embed size)
        ffm_out = ffm_out.view(batch_size, -1)

        # reshape linear_out to (batch size, number of fields)
        linear_out = linear_out.view(batch_size, -1)

        # repeat and reshape bias to shape = (batch size, 1)
        if self.bias is not None:
            bias = self.bias.repeat(batch_size).view(batch_size, 1)

        # cat in dim = 1
        # shape = (batch size, number of fields * embed size + number of fields + 1)
        outputs = torch.cat([ffm_out, linear_out, self.bias], dim=1)

        # fully-connected dense layer for output, return (batch size, output size)
        # or sum with second dimension, return (batch size, 1)
        outputs = self.fc(outputs)

        return outputs
