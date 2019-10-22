import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor

class CrossNetworkLayer(nn.Module):
    r"""Layer class of Cross Network used in Deep & Cross Network :title:`Ruoxi Wang et al, 2017`[1], to 
    calculate cross features interaction between element, by the following equation: for i-th layer, 
    math:`x_{i} = x_{0} * (w_{i} * x_{i-1} + b_{i}) + x_{0}`_.

    :Reference:

    #. `Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`_.
    
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
        
        Raises:
            ValueError: when embed_size or num_fields is missing if using embed_size and num_field pairs, or when inputs_size is missing if using inputs_size
        """
        # refer to parent class
        super(CrossNetworkLayer, self).__init__()

        # bind inputs_size to inputs_size
        self.inputs_size = inputs_size

        # initialize module list for Cross Network
        self.model = nn.ModuleList()

        # add modules to module list of Cross Network
        for _ in range(num_layers):
            self.model.append(nn.Linear(inputs_size, inputs_size))
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """Forward calculation of CrossNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, N, E), dtype = torch.float: Output of CrossNetworkLayer
        """
        # reshape inputs from (B, N, E) to (B, N * E)
        ## emb_inputs = emb_inputs.view(-1, self.inputs_size)
        ## emb_inputs = emb_inputs.flatten(["N", "E"], "E")

        # copy emb_inputs to outputs for residual
        outputs = emb_inputs.detach().requires_grad_()

        # forward calculation of bilinear and add residual
        for layer in self.model:
            # shape = (B, N, E)
            outputs = emb_inputs * layer(outputs) + emb_inputs
        
        # rename tensor names
        if outputs.dim() == 2:
            outputs.names = ("B", "O")
        elif outputs.dim() == 3:
            outputs.names = ("B", "N", "O")

        # unsqueeze outputs to (B, 1, O)
        ## outputs = outputs.unsqueeze(1)

        return outputs
