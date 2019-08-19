from . import _CtrModel
from torecsys.layers import GeneralizedMatrixFactorization, MultilayerPerceptronLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Callable, Typing


class NeuralCollaborativeFiltering(_CtrModel):
    r"""
    """
    def __init__(self, 
                 deep_output_size: int,
                 deep_layer_sizes: int,
                 deep_dropout_p  : List[float] = None,
                 deep_activation : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        
        self.mlp = MultilayerPerceptronLayer(deep_output_size, deep_layer_sizes, deep_dropout_p, deep_activation)
        self.glm = GeneralizedMatrixFactorization()

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        # mlp_out's input shape = () and output shape = ()
        mlp_out = self.mlp(emb_inputs)

        # glm_out's input shape = () and output shape = ()
        glm_out = self.glm(emb_inputs)

        # model's output
        outputs = glm_out + mlp_out
        
        return outputs
    