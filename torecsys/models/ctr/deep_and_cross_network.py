from . import _CtrModel
from torecsys.layers import CrossNetworkLayer, MultilayerPerceptronLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


class DeepAndCrossNetworkModel(_CtrModel):
    r"""DeepAndCrossNetworkModel is a model of deep and cross network, which is a model of 
    a concatenation of deep neural network and cross network, and finally pass to a fully 
    connect layer for the output.

    :Reference:

    #. `Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`_.
    
    """
    def __init__(self, 
                 inputs_size      : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 cross_num_layers : int,
                 output_size      : int = 1,
                 deep_dropout_p   : List[float] = None, 
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""initialize deep adn cross network
        
        Args:
            inputs_size (int): size of inputs tensor
            deep_output_size (int): size of outputs of deep neural network
            deep_layer_sizes (List[int]): sizes of layers in deep neural network
            cross_num_layers (int): number of layers in cross network
            output_size (int, optional): output size of model, i.e. output size of the last fully-connect layer. Defaults to 1.
            deep_dropout_p (List[float], optional): dropout probability for each layer after dense layer. Defaults to None.
            deep_activation (Callable[[T], T], optional): activation function for each layer of deep neural network. Defaults to nn.ReLU().
        """
        super(DeepAndCrossNetworkModel, self).__init__()

        # initialize the layers of module
        # deep output's shape = (B, 1, O_d)
        self.deep = MultilayerPerceptronLayer(output_size=deep_output_size, layer_sizes=deep_layer_sizes, inputs_size=inputs_size, dropout_p=deep_dropout_p, activation=deep_activation)
        # cross output's shape = (B, 1, E)
        self.cross = CrossNetworkLayer(num_layers=cross_num_layers, inputs_size=inputs_size)

        # initialize output fc layer
        cat_size = deep_output_size + inputs_size
        self.fc = nn.Linear(cat_size, output_size)
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of deep and cross network
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: second order terms of fields that will be passed into afm layer and can be get from nn.Embedding(embed_size=E)
        
        Returns:
            T, shape = (B, O), dtype = torch.float: output of deep and cross network
        """
        # inputs' shape = (B, N, I) and reshape to (B, 1, O_d)
        deep_out = self.deep(emb_inputs)

        # inputs' shape = (B, N, I) and cross_out's shape = (B, 1, I)
        cross_out = self.cross(emb_inputs)
        
        # cat in third dimension and return shape = (B, 1, O_d + I)
        # then squeeze() to shape = (B, O_d + I)
        outputs = torch.cat([cross_out, deep_out], dim=2).squeeze()

        # pass outputs to fully-connect layer, and return shape = (B, O)
        outputs = self.fc(outputs)
        
        return outputs
