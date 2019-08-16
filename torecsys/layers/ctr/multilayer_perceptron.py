from torecsys.utils.logging.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Callable, List


class MultilayerPerceptronLayer(nn.Module):
    r"""MultilayerPerceptron is a Fully Connected Feed Forward Neural Network, 
    which is also called Dense Layer, Deep Neural Network etc, to learn high-order
    interaction between different features by element-wise.
    """
    @jit_experimental
    def __init__(self, 
                 output_size : int,
                 layer_sizes : List[int],
                 embed_size  : int = None,
                 num_fields  : int = None,
                 inputs_size : int = None,
                 dropout_p   : List[float] = None,
                 activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        """initialize multilayer perceptron module
        
        Args:
            output_size (int): output size of multilayer perceptron layer
            layer_sizes (List[int]): layer sizes of multilayer perceptron layer
            embed_size (int, optional): embedding size, must input with num_fields together. Defaults to None.
            num_fields (int, optional): number of fields in inputs, must input with embed_size together. Defaults to None.
            inputs_size (int, optional): inputs size, cannot input with embed_size and num_fields. Defaults to None.
            dropout_p (List[float], optional): dropout probability after activation of each layer. Allow: [None, list of float for each layer]. Defaults to None.
            activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function of each layer. Allow: [None, Callable[[torch.Tensor], torch.Tensor]]. Defaults to nn.ReLU().
        
        Raises:
            ValueError: when embed_size or num_fields is missing if using embed_size and num_field pairs, or when inputs_size is missing if using inputs_size
            ValueError: when dropout_p is not None and length of dropout_p is not equal to that of layer_sizes
        """
        # initialize nn.Module class
        super(MultilayerPerceptronLayer, self).__init__()

        # check if length of dropout_p is not equal to length of layer_sizes
        if dropout_p is not None and len(dropout_p) != len(layer_sizes):
            raise ValueError("length of dropout_p must be equal to length of layer_sizes.")
        
        # build the model with nn.Sequential
        self.model = nn.Sequential()
        if inputs_size is None and embed_size is not None and num_fields is not None:
            inputs_size = embed_size * num_fields
        elif inputs_size is not None and (embed_size is None or num_fields is None):
            inputs_size = inputs_size
        else:
            raise ValueError("Only allowed:\n    1. embed_size and num_fields is not None, and inputs_size is None\n    2. inputs_size is not None, and embed_size or num_fields is None")
        layer_sizes = [inputs_size] + layer_sizes
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.model.add_module("linear_%s" % i, nn.Linear(in_f, out_f))
            if activation is not None:
                self.model.add_module("activation_%s" % i, activation)
            if dropout_p is not None:
                self.model.add_module("dropout_%s" % i, nn.Dropout(dropout_p[i]))
        self.model.add_module("linear_output", nn.Linear(layer_sizes[-1], output_size))
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed-forward calculation of multilayer perceptron
        
        Args:
            inputs (torch.Tensor), shape = (B, N, E), dtype = torch.float: features vectors of inputs
        
        Returns:
            torch.Tensor, shape = (B, 1, E), dtype = torch.float: output of multilayer perceptron
        """
        batch_size = inputs.size(0)
        outputs = inputs.view(batch_size, -1)
        outputs = self.model(outputs)
        return outputs.unsqueeze(1)
