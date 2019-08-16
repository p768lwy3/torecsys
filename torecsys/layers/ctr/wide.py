from torecsys.utils.logging.decorator import jit_experimental
import torch
import torch.nn as nn

class WideLayer(nn.Module):
    r"""WideLayer is a (Linear-Dropout) layer frequently used to calculate linear relation.
    """
    @jit_experimental
    def __init__(self,
                 embed_size  : int,
                 num_fields  : int,
                 output_size : int,
                 dropout_p   : float = 0.0):
        r"""initialize wide layer module
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            output_size (int): output size of linear layer
            dropout_p (float, optional): dropout probability after Linear layer. Defaults to 0.0.
        """
        super(WideLayer, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("linear_0", nn.Linear(embed_size * num_fields, output_size))
        self.model.add_module("dropout_0", nn.Dropout(dropout_p))
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed-forward calculation of wide layer
        
        Args:
            inputs (torch.Tensor), shape = (batch size, number of fields, embedding size), dtype = torch.float: features matrices of inputs
        
        Returns:
            torch.Tensor, shape = (batch size, 1, output size), dtype = torch.float: output of wide layer
        """
        # flatten inputs tensor from (batch size, number of fields, embedding size)
        # to (batch size, num of fields * embedding size)
        batch_size = inputs.size(0)
        outputs = inputs.view(batch_size, -1)

        # calculate outputs of model and unsqueeze 2-nd dimension to make outputs' shape be (batch size, 1, output size)
        outputs = self.model(outputs)
        return outputs.unsqueeze(1)
    