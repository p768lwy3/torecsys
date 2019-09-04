from . import _CtrModel
from torecsys.layers import FMLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Callable, List

class NeuralFactorizationMachineLayer(_CtrModel):
    r"""Model class of Neural Factorization Machine (NFM) to pool the embedding tensors from 
    (B, N, E) to (B, 1, E) with Factorization Machine (FM) as an inputs of Deep Neural Network, 
    i.e. a stack of FM and DNN models.

    :Reference:

    #. `Xiangnan He et al, 2017. Neural Factorization Machines for Sparse Predictive Analytics <https://arxiv.org/abs/1708.05027>`_.

    """
    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 fm_dropout_p     : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize Neural Factorization MachineLayer.
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            deep_output_size (int): Output size of DNN
            deep_layer_sizes (List[int]): Layer sizes of DNN
            fm_dropout_p (float, optional): Probability of Dropout in FM. 
                Defaults to 0.0.
            deep_dropout_p (List[float], optional): Probability of Dropout in DNN. 
                Allow: [None, list of float for each layer]. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of Linear. 
                Allow: [None, Callable[[T], T]]. 
                Defaults to nn.ReLU().
        """
        # refer to parent class
        super(NeuralFactorizationMachineLayer, self).__init__()

        # initialize sequential of model
        self.sequential = nn.Sequential()

        # add modules to model
        self.sequential.add_module("b_interaction", FMLayer(fm_dropout_p))
        self.sequential.add_module("hidden", DNNLayer(
            output_size = deep_output_size,
            layer_sizes = deep_layer_sizes,
            inputs_size = cat_size,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        ))

        # initialize bias variable
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.bias.data)
    
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of NeuralFactorizationMachineLayer
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: Features tensors.
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: Output of NeuralFactorizationMachineLayer
        """

        # calculate sequential part, with emb_inputs and outputs' shape = (B, 1, O)
        outputs = self.sequential(emb_inputs)

        # sum all values to the outputs
        outputs = outputs.squeeze().sum(dim=1, keepdim=True)
        outputs = outputs + feat_inputs.squeeze().sum(dim=1, keepdim=True)
        outputs = outputs + self.bias

        return outputs
