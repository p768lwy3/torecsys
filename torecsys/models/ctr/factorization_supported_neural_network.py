from . import _CtrModel
from torecsys.layers import FactorizationMachineLayer, MultilayerPerceptronLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


class FactorizationMachineSupportedNeuralNetwork(_CtrModel):
    r"""FactorizationMachineSupportedNeuralNetwork is a model of Factorization-machine supported Neural
    Network, which is a stack of Factorization Machine and Deep Neural Network, with the following calculation: 
    First calculate features interactions by factorization machine: :math:`y_{FM} = \text{Sigmoid} ( w_{0} + \sum_{i=1}^{N} w_{i} x_{i} + \sum_{i=1}^{N} \sum_{j=i+1}^{N} <v_{i}, v_{j}> x_{i} x_{j} )` .
    Then feed the interactions' representation to deep neural network: :math:`y_{i} = \text{Activation} ( w_{i} y_{i - 1} + b_{i} )` , 
    where :math:`y_{0} = y_{FM}` for the inputs of the first layer in deep neural network.

    :Reference:

    #. `Weinan Zhang et al, 2016. Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction <https://arxiv.org/abs/1601.02376>`_.
    
    """
    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 fm_dropout_p     : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""initialize Factorization-machine Supported Neural Network
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            deep_output_size (int): output size of multilayer perceptron layer
            deep_layer_sizes (List[int]): layer sizes of multilayer perceptron layer
            fm_dropout_p (float, optional): dropout probability after factorization machine. Defaults to 0.0.
            deep_dropout_p (List[float], optional): dropout probability after activation of each layer. Allow: [None, list of float for each layer]. Defaults to None.
            deep_activation (Callable[[T], T], optional): activation function of each layer. Allow: [None, Callable[[T], T]]. Defaults to nn.ReLU().
        """
        super(FactorizationMachineSupportedNeuralNetwork, self).__init__()

        # initialize factorization machine layer
        self.fm = FactorizationMachineLayer(fm_dropout_p)
        
        # initialize dense layers
        cat_size = num_fields + embed_size
        self.deep = MultilayerPerceptronLayer(
            output_size = deep_output_size,
            layer_sizes = deep_layer_sizes,
            inputs_size = cat_size,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        )

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of Factorization-machine Supported Neural Network
        
        Args:
            feat_inputs (T), shape = (B, N, 1): first order outputs, i.e. outputs from nn.Embedding(V, 1)
            emb_inputs (T), shape = (B, N, E): second order outputs
        
        Returns:
            T, shape = (B, O), dtype = torch.float: outputs of Factorization-machine Supported Neural Network Model
        """

        # squeeze feat_inputs to shape = (B, N)
        fm_first = feat_inputs.squeeze()

        # pass to fm layer where its returns' shape = (B, E)
        fm_second = self.fm(emb_inputs).squeeze()

        # cat into a tensor with shape = (B, N + E)
        fm_out = torch.cat([fm_first, fm_second], dim=1)

        # feed-forward to deep neural network, return shape = (B, O)
        outputs = self.deep(fm_out)

        return outputs
    