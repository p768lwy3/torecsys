from . import _CtrModel
from torecsys.layers import FieldAwareFactorizationMachineLayer, MultilayerPerceptronLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import List


class FieldAwareNeuralFactorizationMachineModel(_CtrModel):
    r"""
    """
    def __init__(self, 
                 embed_size       : int,
                 num_fields       : int,
                 output_size      : int,
                 layers_size      : List[int],
                 ffm_dropout_p    : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor, torch.Tensor]] = nn.ReLU()):
        r"""initialize Field-aware Neural Factorization Machine Model
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            output_size (int): output size of multilayer perceptron layer
            layers_size (List[int]): layer sizes of multilayer perceptron layer
            ffm_dropout_p (float, optional): dropout probability after ffm layer. Defaults to 0.0.
            deep_dropout_p (List[float], optional): dropout probability after each mlp layer. Allow: [None, List[float]]. Defaults to None.
            deep_activation (Callable[[T, T]], optional): activation after each mlp layer. Allow: [None, Callable[[T], T]]. Defaults to nn.ReLU().
        """
        # initialize nn.Module class
        super(FieldAwareNeuralFactorizationMachineModel, self).__init__()

        # initialize ffm layer
        self.ffm = FieldAwareFactorizationMachineLayer(ffm_dropout_p)

        # initialize dense layers
        self.cat_size = num_fields * embed_size
        self.deep = MultilayerPerceptronLayer(
            output_size = output_size, 
            layer_sizes = layers_sizes,
            inputs_size = num_fields + self.cat_size,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        )
    
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""[summary]
        
        Args:
            feat_inputs (torch.Tensor): [description]
            emb_inputs (torch.Tensor): [description]
        
        Returns:
            torch.Tensor: [description]
        """
        # squeeze feat_inputs to shape = (B, N)
        ffm_first = feat_inputs.squeeze()

        # calculate bi-interaction vectors by ffm and return shape = (B, N, E)
        # then, reshape bi-interaction vectors to shape = (B, N * E)
        ffm_second = self.ffm(emb_inputs).squeeze().view(-1, self.cat_size)

        # concatenate linear terms and second-order terms before mlp layer
        ffm_out = torch.cat(ffm_first, ffm_second)

        # feed-forward to deep neural network, return shape = (B, O)
        ouputs = self.deep(ffm_outputs)

        return outputs
