from . import _CtrModel
from torecsys.layers import FMLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn
from typing import Callable, List

class NeuralFactorizationMachineModel(_CtrModel):
    r"""Model class of Neural Factorization Machine (NFM).
    
    Neural Factorization Machine is a model to pool embedding tensors from (B, N, E) to (B, 1, E) 
    with Factorization Machine (FM) as an inputs of Deep Neural Network, i.e. a stack of 
    factorization machine and dense network.

    :Reference:

    #. `Xiangnan He et al, 2017. Neural Factorization Machines for Sparse Predictive Analytics <https://arxiv.org/abs/1708.05027>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size       : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 fm_dropout_p     : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize NeuralFactorizationMachineModel.
        
        Args:
            embed_size (int): Size of embedding tensor
            deep_output_size (int): Output size of dense network
            deep_layer_sizes (List[int]): Layer sizes of dense network
            fm_dropout_p (float, optional): Probability of Dropout in FM. 
                Defaults to 0.0.
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network. 
                Defaults to nn.ReLU().
        
        Attributes:
            sequential (nn.Sequential): Module of sequential moduels, including factorization
                machine layer and dense layer.
            bias (nn.Parameter): Parameter of bias of output projection.
        """
        # refer to parent class
        super(NeuralFactorizationMachineModel, self).__init__()

        # initialize sequential module
        self.sequential = nn.Sequential()

        # initialize fm layer
        self.sequential.add_module("B_interaction", FMLayer(fm_dropout_p))

        # initialize dense layer
        self.sequential.add_module("Deep", DNNLayer(
            output_size = deep_output_size,
            layer_sizes = deep_layer_sizes,
            inputs_size = embed_size,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        ))

        # initialize bias parameter
        self.bias = nn.Parameter(torch.zeros((1, 1), names=("B", "O")))
        nn.init.uniform_(self.bias.data)
    
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of NeuralFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: Features tensors.
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: Output of NeuralFactorizationMachineModel.
        """
        # Aggregate feat_inputs on dimension N and rename dimesion E to O
        # inputs: feat_inputs, shape = (B, N, E)
        # output: nfm_first, shape = (B, O = 1)
        nfm_first = feat_inputs.sum(dim="N").rename(E="O")

        # Calculate with sequential module forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: nfm_second, shape = (B, O)
        nfm_second = self.sequential(emb_inputs)

        # Add up nfm_second, nfm_first and bias
        # inputs: nfm_second, shape = (B, O = 1)
        # inputs: nfm_first, shape = (B, O = 1)
        # inputs: bias, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = nfm_second + nfm_first + self.bias

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
