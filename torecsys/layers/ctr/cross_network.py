from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn

class CrossNetworkLayer(nn.Module):
    r"""CrossNetworkLayer is a layer used in Deep & Cross Network to calculate the low dimension 
    element-wise cross-feature interaction by the following equation:
    :math:`\text{for i-th layer, } \bm{x}_{i} = x_{0} * (w_{i} * x_{i-1} + b_{i}) + x_{0}` .

    :Reference:

    #. `Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`_.

    """
    @jit_experimental
    def __init__(self, 
                 num_layers: int,
                 embed_size  : int = None,
                 num_fields  : int = None,
                 inputs_size : int = None):
        r"""initialize cross network layer module
        
        Args:
            num_layers (int): number of layers of cross network layer
            embed_size (int, optional): embedding size, must input with num_fields together. Defaults to None.
            num_fields (int, optional): number of fields in inputs, must input with embed_size together. Defaults to None.
            inputs_size (int, optional): inputs size, cannot input with embed_size and num_fields. Defaults to None.
        
        Raises:
            ValueError: when embed_size or num_fields is missing if using embed_size and num_field pairs, or when inputs_size is missing if using inputs_size
        """
        # initialize nn.Module class
        super(CrossNetworkLayer, self).__init__()
        if inputs_size is None and embed_size is not None and num_fields is not None:
            inputs_size = embed_size * num_fields
        elif inputs_size is not None and (embed_size is None or num_fields is None):
            inputs_size = inputs_size
        else:
            raise ValueError("Only allowed:\n    1. embed_size and num_fields is not None, and inputs_size is None\n    2. inputs_size is not None, and embed_size or num_fields is None")

        # initialize cross network module list
        self.model = nn.ModuleList()
        for _ in range(num_layers):
            self.model.append(nn.Linear(inputs_size, inputs_size))
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """feed-forward calculation of cross network layer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: features matrices of inputs
        
        Returns:
            T, shape = (B, 1, E), dtype = torch.float: output of cross network layer
        """
        # reshape inputs from 3 dimensiol to 2 dimension with shape = (B, N * E)
        batch_size = emb_inputs.size(0)
        emb_inputs = emb_inputs.view(batch_size, -1)

        # copy a tensor call outpus from inputs
        outputs = emb_inputs.detach().requires_grad_()

        for layer in self.model:
            outputs = emb_inputs * layer(outputs) + emb_inputs
        
        return outputs.unsqueeze(1)
