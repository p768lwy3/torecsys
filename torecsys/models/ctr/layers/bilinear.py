import torch
import torch.nn as nn


class BilinearNetworkLayer(nn.Module):
    r"""BilinearNetworkLayer is a bilinear network layer to calculate the low dimension 
    element-wise feature interaction by nn.Bilinear function:
    :math:`\text{for i-th layer, } \bm{x}_{i} = (x_{0} * A_{i} * x_{i - 1}) + b_{i} + x_{0}`_.
    """
    def __init__(self,
                 embed_size  : int, 
                 num_fields  : int, 
                 output_size : int, 
                 num_layers  : int):
        r"""initialize bilinear network layer module

        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            output_size (int): output size of bilinear network layer
            num_layers (int): number of layers of bilinear network layer
        """
        # initialize nn.Module class
        super(BilinearNetworkLayer, self).__init__()
        inputs_size = embed_size * num_fields

        # initialize bilinear network module list
        self.model = nn.ModuleList()
        for _ in range(num_layers):
            self.model.append(nn.Bilinear(inputs_size, inputs_size, inputs_size))
        self.fc = nn.Linear(inputs_size, output_size)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """feed-forward calculation of bilinear network layer
        
        Args:
            inputs (torch.Tensor), shape = (B, N, E), dtype = torch.float: features matrices of inputs
        
            Returns:
                torch.Tensor, shape = (batch size, 1, output size), dtype = torch.float: output of bilinear network layer
            """
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1)
        outputs = inputs.detach().requires_grad_()

        for layer in self.model:
            outputs = layer(inputs, outputs) + inputs
        
        output = self.fc(outputs)
        return outputs.unsqueeze(1)
