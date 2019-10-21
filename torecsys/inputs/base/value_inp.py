from . import _Inputs
from collections import namedtuple
import torch
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor


class ValueInputs(_Inputs):
    r"""Base Inputs class for value to be passed directly.
    
    :Todo:

    #. add transforms for value inputs to do preprocessing

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, num_fields: int):
        r"""Initialize ValueInputs
        
        Args:
            num_fields (int): Number of inputs' fields.

        Attributes:
            length (int): Number of inputs' fields.
        """
        # refer to parent class
        super(ValueInputs, self).__init__()

        # bind length to length of inp_fields 
        self.length = num_fields
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of ValueInputs.
        
        Args:
            inputs (T), shape = (B, N): Tensor of values in input fields.
        
        Returns:
            T, shape = (B, N): Outputs of ValueInputs
        """
        inputs.names = ("B", "E")
        return inputs
