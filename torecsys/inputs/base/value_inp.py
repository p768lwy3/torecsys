from . import _Inputs
from torecsys.utils.decorator import jit_experimental
import torch

class ValueInputs(_Inputs):
    r"""Base Inputs class for value to be passed directly.
    
    :Todo:

    #. add transforms for value inputs to do preprocessing

    """
    @jit_experimental
    def __init__(self, num_fields: int):
        r"""Initialize ValueInputs
        
        Args:
            num_fields (int): Number of inputs' fields.

        Attributes:
            length (int): Number of inputs' fields.
        """
        # refer to parent class
        super(ValueInputs, self).__init__()

        # bind length to num_fields 
        self.length = num_fields
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of ValueInputs.
        
        Args:
            inputs (T), shape = (B, N): Tensor of values in input fields.
        
        Returns:
            T, shape = (B, 1, N): Outputs of ValueInputs
        """
        # unsqueeze(1) and return
        return inputs.unsqueeze(1)
