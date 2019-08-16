from . import _CtrModule
from ..layers import CrossNetworkLayer, MultilayerPerceptronLayer
from torecsys.utils.logging.decorator import jit_experimental
import torch
import torch.nn as nn


class DeepAndCrossNetworkModule(_CtrModule):
    r"""
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
            inputs_size (int): [description]
            deep_output_size (int): [description]
            deep_layer_sizes (List[int]): [description]
            cross_num_layers (int): [description]
            output_size (int, optional): [description]. Defaults to 1.
            deep_dropout_p (List[float], optional): [description]. Defaults to None.
            deep_activation (Callable[[torch.Tensor], torch.Tensor], optional): [description]. Defaults to nn.ReLU().
        """
        super(DeepAndCrossNetworkModule, self).__init__()

        # initialize the layers of module
        # deep output's shape = (batch size, 1, output size of deep)
        self.deep = MultilayerPerceptronLayer(output_size=deep_output_size, layer_sizes=deep_layer_sizes, inputs_size=inputs_size, dropout_p=deep_dropout_p, activation=deep_activation)
        # cross output's shape = (batch size, 1, embedding size)
        self.cross = CrossNetworkLayer(num_layers=cross_num_layers, inputs_size=inputs_size)

        # initialize output fc layer
        cat_size = deep_output_size + inputs_size
        self.fc = nn.Linear(cat_size, output_size)
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of deep and cross network
        
        Args:
            emb_inputs (torch.Tensor), shape = (batch size, number of fields, embed size), dtype = torch.float: second order terms of fields that will be passed into afm layer and can be get from nn.Embedding(embed_size=embed_size)
        
        Returns:
            torch.Tensor: output of deep and cross network
        """
        # inputs' shape = (batch size, num of fields, inputs size)
        # deep_out's shape = (batch size, 1, deep_output_size)
        deep_out = self.deep(emb_inputs)

        # inputs' shape = (batch size, num of fields, inputs size)
        # cross_out's shape = (batch size, 1, inputs size)
        cross_out = self.cross(emb_inputs)
        
        # cat in third dimension and return shape = (batch size, 1, deep_output_size + inputs_size)
        # then squeeze() to shape = (batch size, deep_output_size + inputs_size)
        outputs = torch.cat([cross_out, deep_out], dim=2).squeeze()

        # pass outputs to fully-connect layer, and return shape = (batch size, output size)
        outputs = self.fc(outputs)
        
        return outputs
