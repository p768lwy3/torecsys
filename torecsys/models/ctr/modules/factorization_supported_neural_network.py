from . import _CtrModule
from ..layers import FactorizationMachineLayer, MultilayerPerceptronLayer
from torecsys.utils.logging.decorator import jit_experimental
import torch
import torch.nn as nn

class FactorizationMachineSupportedNeuralNetwork(_CtrModule):
    r"""FactorizationMachineSupportedNeuralNetwork is a module of Factorization-machine supported Neural
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
        """initialize Factorization-machine Supported Neural Network
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            deep_output_size (int): output size of multilayer perceptron layer
            deep_layer_sizes (List[int]): layer sizes of multilayer perceptron layer
            fm_dropout_p (float, optional): dropout probability after factorization machine. Defaults to 0.0.
            deep_dropout_p (List[float], optional): dropout probability after activation of each layer. Allow: [None, list of float for each layer]. Defaults to None.
            deep_activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function of each layer. Allow: [None, Callable[[torch.Tensor], torch.Tensor]]. Defaults to nn.ReLU().
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

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""feed forward of Factorization-machine Supported Neural Network
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of inputs torch.Tensor
        
        Key-Values:
            first_order, shape = (batch size, num_fields, 1): first order outputs, i.e. outputs from nn.Embedding(vocab_size, 1)
            second_order, shape = (batch size, num_fields, embed_size): second order outputs
        
        Returns:
            torch.Tensor, shape = (batch size, output size), dtype = torch.float: outputs of Factorization-machine Supported Neural Network Module
        """
        # get batch size
        batch_size = inputs["first_order"].size(0)

        # first_order's shape = (batch size, number of fields, 1)
        # and the output's shape = (batch size, number of fields)
        first_out = inputs["first_order"]
        first_out = first_out.view(batch_size, -1)

        # second_order's shape = (batch size, number of fields * number of fields, embed size)
        # and the output's shape = (batch size, output size)
        second_out = self.fm(inputs["second_order"])
        second_out = second_out.view(batch_size, -1)

        # cat and feed-forward to Dense-layers
        outputs = torch.cat([first_out, second_out], dim=1)
        outputs = self.deep(outputs)

        return outputs
    