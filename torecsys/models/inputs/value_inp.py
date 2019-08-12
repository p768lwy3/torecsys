from . import _Inputs
import torch

class ValueInputs(_Inputs):
    r"""ValueInputs is a input field to pass the value directly
    
    TODO:
    #. add transforms for value inputs to do preprocessing
    """
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
            inputs (torch.Tensor), shape = (batch size, num fields): inputs value tensor
        
        Returns:
            torch.Tensor, shape = (batch size, 1, number of fields): reshaped inputs value tensor
        """
        return inputs.unsqueeze(1)
