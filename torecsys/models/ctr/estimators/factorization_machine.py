from . import _CtrEstimator
from ..layers import FactorizationMachineLayer, WideLayer

import torch
import torch.nn as nn

class FactorizationMachine(_CtrEstimator):
    r"""
    """
    def __init__(self, 
                 embed_size      : int,
                 num_fields      : int,
                 fm_dropout_p    : float,
                 wide_output_size: int, 
                 wide_dropout_p  : float,
                 output_method   : str,
                 output_size     : int):
        
        super(FactorizationMachine, self).__init__()
        
        self.fm = FactorizationMachineLayer(fm_dropout_p)
        self.wide = WideLayer(embed_size, num_fields, wide_output_size, wide_dropout_p)
        
        if output_method == "concatenate":
            self.output_size = embed_size + wide_output_size
        else:
            self.output_size = embed_size
        self.output_method = output_method
        self.fc = nn.Linear(self.output_size, output_size)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        fm_out = self.fm(inputs)
        wide_out = self.wide(inputs)

        if self.output_method == "concatenate":
            outputs = torch.cat([fm_out, wide_out], dim=1)
        elif self.output_method == "dot":
            outputs = fm_out * wide_out
        elif self.output_method == "sum":
            outputs = fm_out + wide_out
        
        outputs = self.fc(outputs.view(-1, self.output_size))
        return outputs
