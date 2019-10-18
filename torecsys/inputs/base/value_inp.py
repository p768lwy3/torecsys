from . import _Inputs
from collections import namedtuple
import torch
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import List

class ValueInputs(_Inputs):
    r"""Base Inputs class for value to be passed directly.
    
    :Todo:

    #. add transforms for value inputs to do preprocessing

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, inp_fields: List[str]):
        r"""Initialize ValueInputs
        
        Args:
            num_fields (int): Number of inputs' fields.

        Attributes:
            length (int): Number of inputs' fields.
        """
        # refer to parent class
        super(ValueInputs, self).__init__()

        # initialize schema
        self.set_schema(inp_fields)

        # bind length to length of inp_fields 
        self.length = len(inp_fields)
    
    def set_schema(self, inputs: List[str]):
        schema = namedtuple("Schema", ["inputs"])
        self.schema = schema(inputs=inputs)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of ValueInputs.
        
        Args:
            inputs (T), shape = (B, N): Tensor of values in input fields.
        
        Returns:
            T, shape = (B, N): Outputs of ValueInputs
        """
        inputs.names = ("B", "E")
        return inputs
