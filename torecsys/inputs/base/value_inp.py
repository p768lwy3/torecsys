from . import _Inputs
from torecsys.utils.decorator import jit_experimental
import torch

class ValueInputs(_Inputs):
    r"""ValueInputs is a input field to pass the value directly
    
    :Todo:

    #. add transforms for value inputs to do preprocessing

    """
    @jit_experimental
    def __init__(self, num_fields: int):
        r"""initialize the value inputs field
        
        Args:
            num_fields (int): total number of fields of inputs
        """
        super(ValueInputs, self).__init__()
        self.num_fields = num_fields
        self.length = num_fields
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Return values of fields
        
        Args:
            inputs (T), shape = (B, N): inputs value tensor
        
        Returns:
            T, shape = (B, 1, N): reshaped inputs value tensor
        """
        return inputs.unsqueeze(1)
