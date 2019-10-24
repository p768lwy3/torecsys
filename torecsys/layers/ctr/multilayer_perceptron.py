import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Callable, List

class MultilayerPerceptronLayer(nn.Module):
    r"""Layer class of Multilayer Perceptron (MLP), which is also called fully connected 
    layer, dense layer, deep neural network, etc, to calculate high order non linear 
    relations of features with a stack of linear, dropout and activation.
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 inputs_size : int,
                 output_size : int,
                 layer_sizes : List[int],
                 dropout_p   : List[float] = None,
                 activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        """Initialize MultilayerPerceptronLayer
        
        Args:
            inputs_size (int): Input size of MLP, i.e. size of embedding tensor. 
            output_size (int): Output size of MLP
            layer_sizes (List[int]): Layer sizes of MLP
            dropout_p (List[float], optional): Probability of Dropout in MLP. 
                Defaults to None.
            activation (Callable[[T], T], optional): Activation function in MLP. 
                Defaults to nn.ReLU().
        
        Attributes:
            inputs_size (int): Input size of MLP. 
            model (torch.nn.Sequential): Sequential of MLP.
        
        Raises:
            ValueError: when embed_size or num_fields is missing if using embed_size and num_field pairs, or when inputs_size is missing if using inputs_size
            ValueError: when dropout_p is not None and length of dropout_p is not equal to that of layer_sizes
        """
        # Refer to parent class
        super(MultilayerPerceptronLayer, self).__init__()

        # Check if length of dropout_p is not equal to length of layer_sizes
        if dropout_p is not None and len(dropout_p) != len(layer_sizes):
            raise ValueError("length of dropout_p must be equal to length of layer_sizes.")
        
        # Bind inputs_size to inputs_size
        self.inputs_size = inputs_size

        # Create a list of inputs_size and layer_sizes
        layer_sizes = [inputs_size] + layer_sizes

        # Initialize sequential of model
        self.model = nn.Sequential()  
        
        # Initialize module of linear, activation and dropout, and add them to sequential module
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.model.add_module("Linear_%s" % i, nn.Linear(in_f, out_f))
            if activation is not None:
                self.model.add_module("Activation_%s" % i, activation)
            if dropout_p is not None:
                self.model.add_module("Dropout_%s" % i, nn.Dropout(dropout_p[i]))
        
        # Initilaize module of linear and add it to sequential module
        self.model.add_module("LinearOutput", nn.Linear(layer_sizes[-1], output_size))
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of MultilayerPerceptronLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, N, O), dtype = torch.float: Output of MLP.
        """
        # Calculate with model forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, O)
        outputs = self.model(emb_inputs.rename(None))

        # Rename tensor names
        if outputs.dim() == 2:
            outputs.names = ("B", "O")
        elif outputs.dim() == 3:
            outputs.names = ("B", "N", "O")

        return outputs
