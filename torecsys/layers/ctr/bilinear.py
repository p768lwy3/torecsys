import torch
import torch.nn as nn

from torecsys.utils.decorator import no_jit_experimental_by_namedtensor


class BilinearNetworkLayer(nn.Module):
    r"""Layer class of Bilinear. 
    
    Bilinear is to calculate interaction in element-wise by nn.Bilinear, which the calculation
    is: for i-th layer, :math:`x_{i} = (x_{0} * A_{i} * x_{i - 1}) + b_{i} + x_{0}`, where 
    :math:`A_{i}` is the weight of module of shape :math:`(O_{i}, I_{i1}, I_{i2})`.

    """

    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 inputs_size: int,
                 num_layers: int):
        r"""Initialize BilinearNetworkLayer

        Args:
            inputs_size (int): Input size of Bilinear, i.e. size of embedding tensor. 
            num_layers (int): Number of layers of Bilinear Network
        
        Attributes:
            inputs_size (int): Size of inputs, or Product of embed_size and num_fields.
            model (torch.nn.ModuleList): Module List of Bilinear Layers.
        """
        # Refer to parent class
        super(BilinearNetworkLayer, self).__init__()

        # Bind inputs_size to inputs_size
        self.inputs_size = inputs_size

        # Initialize module list for Bilinear
        self.model = nn.ModuleList()

        # Initialize bilinear layers and add them to module list
        for _ in range(num_layers):
            self.model.append(nn.Bilinear(inputs_size, inputs_size, inputs_size))

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """Forward calculation of BilinearNetworkLayer
        
        Args:
            emb_inputs (T), shape = shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, N, E), dtype = torch.float: Output of BilinearNetworkLayer.
        """
        # Deep copy emb_inputs to outputs as residual
        # inputs: emb_inputs, shape = (B, O = N * E)
        # output: outputs, shape = (B, O = N * E)
        outputs = emb_inputs.detach().requires_grad_()

        # Calculate with bilinear forwardly and add residual to outputs
        # inputs: emb_inputs, shape = (B, N, E)
        # inputs: outputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E)
        for layer in self.model:
            outputs = layer(emb_inputs.rename(None), outputs.rename(None))
            outputs = outputs + emb_inputs

        # Rename tensor names
        if outputs.dim() == 2:
            outputs.names = ("B", "O")
        elif outputs.dim() == 3:
            outputs.names = ("B", "N", "O")

        return outputs
