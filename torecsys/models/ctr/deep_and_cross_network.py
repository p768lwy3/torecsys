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
                 O_d : int,
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
        # deep output's shape = (B, 1, O_d)
        self.deep = MultilayerPerceptronLayer(output_size=deep_output_size, layer_sizes=deep_layer_sizes, inputs_size=inputs_size, dropout_p=deep_dropout_p, activation=deep_activation)
        # cross output's shape = (B, 1, E)
        self.cross = CrossNetworkLayer(num_layers=cross_num_layers, inputs_size=inputs_size)

        # initialize output fc layer
        cat_size = O_d + inputs_size
        self.fc = nn.Linear(cat_size, output_size)
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of deep and cross network
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: second order terms of fields that will be passed into afm layer and can be get from nn.Embedding(embed_size=E)
        
        Returns:
            torch.Tensor: output of deep and cross network
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
