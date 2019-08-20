from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


# the output shape of this function need to be confirmed
class BilinearNetworkLayer(nn.Module):
    r"""BilinearNetworkLayer is a bilinear network layer to calculate the low dimension 
    element-wise feature interaction by nn.Bilinear function: for i-th layer, 
    :math:`x_{i} = (x_{0} * A_{i} * x_{i - 1}) + b_{i} + x_{0}` .
    """
    @jit_experimental
    def __init__(self,
                 output_size : int, 
                 num_layers  : int,
                 embed_size  : int = None, 
                 num_fields  : int = None,
                 inputs_size : int = None):
        r"""initialize bilinear network layer module

        Args:
            output_size (int): output size of bilinear network layer
            num_layers (int): number of layers of bilinear network layer
            embed_size (int, optional): embedding size, must input with num_fields together. Defaults to None.
            num_fields (int, optional): number of fields in inputs, must input with embed_size together. Defaults to None.
            inputs_size (int, optional): inputs size, cannot input with embed_size and num_fields. Defaults to None.
        
        Raises:
            ValueError: when embed_size or num_fields is missing if using embed_size and num_field pairs, or when inputs_size is missing if using inputs_size
        """
        # initialize nn.Module class
        super(BilinearNetworkLayer, self).__init__()
        if inputs_size is None and embed_size is not None and num_fields is not None:
            inputs_size = embed_size * num_fields
        elif inputs_size is not None and (embed_size is None or num_fields is None):
            inputs_size = inputs_size
        else:
            raise ValueError("Only allowed:\n    1. embed_size and num_fields is not None, and inputs_size is None\n    2. inputs_size is not None, and embed_size or num_fields is None")

        # initialize bilinear network module list
        self.model = nn.ModuleList()
        for _ in range(num_layers):
            self.model.append(nn.Bilinear(inputs_size, inputs_size, inputs_size))
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """feed-forward calculation of bilinear network layer
        
        Args:
            emb_inputs (T), shape = shape = (B, N, E) or (B, 1, I), dtype = torch.float: features matrices of inputs
        
            Returns:
                T, shape = (B, 1, N * E) or (B, 1, I), dtype = torch.float: output of bilinear network layer
        """
        batch_size = emb_inputs.size(0)
        emb_inputs = emb_inputs.view(batch_size, -1)
        outputs = emb_inputs.detach().requires_grad_()

        for layer in self.model:
            # return size = (B, N * E)
            outputs = layer(emb_inputs, outputs) + emb_inputs
        
        return outputs.unsqueeze(1)
