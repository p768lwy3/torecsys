from . import _CtrModule
from ..layers import FactorizationMachineLayer, MultilayerPerceptronLayer
from torecsys.utils.logging.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Dict


class DeepFactorizationMachineModule(_CtrModule):
    r"""DeepFactorizationMachineModule is a module of Deep Factorization Machine (DeepFM) proposed
    by Huawei in 2017, which add up outputs of factorization machine and fully-connected dense 
    neural network directly: :math:`y_{out} = y_{deep} + y_{fm}` , to gain the advantage of two 
    different models of two different objectives, i.e. to gain the explainable power in high dimension 
    of Deep Neural Network, and to gain the explainable power in low dimension of Factorization 
    Machine, Hence, the :math:`y_{deep}` and :math:`y_{fm}` are calculated with the following 
    equations:

    #. for the deep part, :math:`y_{deep}` is the outcome of a Deep Feed-forward Neural Network, 
    which is equal to :math:`y_{i} = \text{activation} ( W_{i} a_{i - 1} + b_{i} )` .

    #. and for the fm part, :math:`y_{fm}` is the result of a factorization machine calculate, 
    which is equal to :math:`y_{fm} = \sum_{i=1}^{n} w_{i} x_{i} + \sum_{i=1}^{n} \sum_{j=1+1}^{n} <v_{i},v_{j}> x_{i} x_{j}` .

    :Reference:

    #. `Huifeng Guo et al, 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction <https://arxiv.org/abs/1703.04247>`_.
    """
    def __init__(self, 
                 embed_size       : int,
                 num_fields       : int,
                 deep_layer_sizes : List[int],
                 fm_dropout_p     : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 output_size      : int = 1):
        r"""initialize Deep Factorization Machine Module
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            deep_layer_sizes (List[int]): layer sizes of multilayer perceptron layer
            fm_dropout_p (float, optional): dropout probability after factorization machine. Defaults to 0.0.
            deep_dropout_p (List[float], optional): dropout probability after activation of each layer. Allow: [None, list of float for each layer]. Defaults to None.
            deep_activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function of each layer. Allow: [None, Callable[[torch.Tensor], torch.Tensor]]. Defaults to nn.ReLU().
            output_size (int, optional): output size of linear transformation after concatenate. Defaults to 1.
        """
        # initialize nn.Module class
        super(DeepFactorizationMachineModule, self).__init__()

        # layers (deep and fm) of second-order part of inputs
        self.fm = FactorizationMachineLayer(fm_dropout_p)
        self.deep = MultilayerPerceptronLayer(
            output_size=1,
            layer_sizes=deep_layer_sizes,
            embed_size=embed_size,
            num_fields=num_fields,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""feed forward of Deep Factorization Machine Module
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of inputs torch.Tensor
        
        Key-Values:
            first_order, shape = (batch size, num_fields, 1): first order outputs, i.e. outputs from nn.Embedding(vocab_size, 1)
            second_order, shape = (batch size, num_fields, embed_size): second order outputs of one-hot encoding, i.e. outputs from nn.Embedding(vocab_size, embed_size)
        
        Returns:
            torch.Tensor, shape = (batch size, output size), dtype = torch.float: outputs of Deep Factorization Machine Module
        """

        # first_order'shape = (batch size, number of fields, 1)
        # and the output's shape = (batch size, number of fields)
        first_out = inputs["first_order"].squeeze()

        # second_order's shape = (batch size, number of fields, embed size)
        second_in = inputs["second_order"]

        # pass to fm layer with shape = (batch size, embed size)
        second_out = self.fm(second_in).squeeze()
        
        # calculate output of factorization machine with output's shape = (batch size, 1)
        fm_out = torch.cat([first_out, second_out], dim=1)
        fm_out = fm_out.sum(dim=1, keepdim=True)

        # pass to dense layers with output's shape = (batch size, 1)
        deep_out = self.deep(second_in)

        # deepfm outputs = fm_out + deep_out
        outputs = deep_out + fm_out
        
        return outputs
