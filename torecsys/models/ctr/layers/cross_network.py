import torch
import torch.nn as nn

class CrossNetworkLayer(nn.Module):
    r"""CrossNetworkLayer is a layer used in Deep & Cross Network to calculate the low dimension 
    element-wise cross-feature interaction by the following equation:
    :math:`\text{for i-th layer, } \bm{x}_{i} = x_{0} * (w_{i} * x_{i-1} + b_{i}) + x_{0}`_.

    Reference:
        `Ruoxi Wang et. al 2017, Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`
    """
    def __init__(self, 
                 embed_size  : int, 
                 num_fields  : int, 
                 output_size : int, 
                 num_layers  : int):
        r"""initialize cross network layer module
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            output_size (int): output size of cross network layer
            num_layers (int): number of layers of cross network layer
        """
        # initialize nn.Module class
        super(CrossNetworkLayer, self).__init__()
        inputs_size = embed_size * num_fields

        # initialize cross network module list
        self.model = nn.ModuleList()
        for _ in range(num_layers):
            self.model.append(nn.Linear(inputs_size, inputs_size))
        self.fc = nn.Linear(inputs_size, output_size)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """feed-forward calculation of cross network layer
        
        Args:
            inputs (torch.Tensor), shape = (B, N, E), dtype = torch.float: features matrices of inputs
        
        Returns:
            torch.Tensor, shape = (batch size, 1, output size), dtype = torch.float: output of cross network layer
        """
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1)
        outputs = inputs.detach().requires_grad_()

        for layer in self.model:
            outputs = inputs * layer(outputs) + inputs
        
        outputs = self.fc(outputs)
        return outputs.unsqueeze(1)
