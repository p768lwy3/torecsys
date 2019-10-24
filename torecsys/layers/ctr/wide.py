import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor

class WideLayer(nn.Module):
    r"""Layer class of wide. 
    
    Wide is a stack of linear and dropout, used in calculation of linear relation frequently.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 inputs_size : int,
                 output_size : int,
                 dropout_p   : float = 0.0):
        r"""Initialize WideLayer
        
        Args:
            inputs_size (int, optional): Size of inputs, i.e. size of embedding tensor.
            output_size (int): Output size of wide layer
            dropout_p (float, optional): Probability of Dropout in wide layer. 
                Defaults to 0.0.
        
        Attributes:
            inputs_size (int): Size of inputs, or Product of embed_size and num_fields.
            model (torch.nn.Sequential): Sequential of wide layer.
        """
        # Refer to parent class
        super(WideLayer, self).__init__()

        # Bind inputs_size to inputs_size
        self.inputs_size = inputs_size

        # Initialize module of sequential
        self.model = nn.Sequential()

        # Initialize linear and dropout and add them to module
        self.model.add_module("Linear", nn.Linear(inputs_size, output_size))
        self.model.add_module("Dropout", nn.Dropout(dropout_p))
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of WideLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, N, E), dtype = torch.float: Output of wide layer.
        """
        # Calculate with linear forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E = O)
        outputs = self.model(emb_inputs.rename(None))

        # Rename tensor names
        if outputs.dim() == 2:
            outputs.names = ("B", "O")
        elif outputs.dim() == 3:
            outputs.names = ("B", "N", "O")

        return outputs
    