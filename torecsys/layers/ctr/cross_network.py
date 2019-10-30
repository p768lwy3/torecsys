import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor

class CrossNetworkLayer(nn.Module):
    r"""Layer class of Cross Network. 
    
    Cross Network was used in Deep & Cross Network, to calculate cross features interaction 
    between element, by the following equation: for i-th layer, :math:`x_{i} = x_{0} * (w_{i} * x_{i-1} + b_{i}) + x_{0}`.

    :Reference:

    #. `Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`
    
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 inputs_size : int,
                 num_layers  : int):
        r"""Initialize CrossNetworkLayer
        
        Args:
            inputs_size (int): Inputs size of Cross Network, i.e. size of embedding tensor.
            num_layers (int): Number of layers of Cross Network
        
        Attributes:
            inputs_size (int): Inputs size of Cross Network.
            model (torch.nn.ModuleList): Module List of Cross Network Layers.
        """
        # Refer to parent class
        super(CrossNetworkLayer, self).__init__()

        # Bind inputs_size to inputs_size
        self.inputs_size = inputs_size

        # Initialize module list for Cross Network
        self.model = nn.ModuleList()

        # Initialize linear layer and add to module list of Cross Network
        for _ in range(num_layers):
            self.model.append(nn.Linear(inputs_size, inputs_size))
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """Forward calculation of CrossNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, N, E), dtype = torch.float: Output of CrossNetworkLayer
        """
        # Deep copy emb_inputs to outputs as residual
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E = O)
        outputs = emb_inputs.detach().requires_grad_()

        # Calculate with linear forwardly and add residual to outputs
        # inputs: emb_inputs, shape = (B, N, E)
        # inputs: outputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E)
        for layer in self.model:
            outputs = emb_inputs * layer(outputs) + emb_inputs
        
        # Rename tensor names
        if outputs.dim() == 2:
            outputs.names = ("B", "O")
        elif outputs.dim() == 3:
            outputs.names = ("B", "N", "O")

        return outputs
