from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import GMFLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel


class NeuralCollaborativeFilteringModel(CtrBaseModel):
    r"""
    Model class of Neural Collaborative Filtering (NCF).
    
    Neural Collaborative Filtering is a model that concatenate dense network and generalized matrix factorization,
    i.e. :math:`y_{mf}(x) = \sum_{v=1}^{k}(x_{u}^{v} * x_{j}^{v})`

    :Reference:

    #. `Xiangnan He, 2017. Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_.

    """

    def __init__(self,
                 embed_size: int,
                 deep_output_size: int,
                 deep_layer_sizes: List[int],
                 deep_dropout_p: List[float] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize NeuralCollaborativeFilteringModel
        
        Args:
            embed_size (int): size of embedding tensor
            deep_output_size (int): output size of dense network
            deep_layer_sizes (List[int]): layer sizes of dense network
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.deep = DNNLayer(
            inputs_size=embed_size * 2,
            output_size=deep_output_size,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

        self.glm = GMFLayer()

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of NeuralCollaborativeFilteringModel
        
        Args:
            emb_inputs (T), shape = (B, N = 2, E), data_type = torch.float:
                embedded features tensors of users and items
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of NeuralCollaborativeFilteringModel
        """
        # Name the emb_inputs tensor for flatten
        emb_inputs.names = ('B', 'N', 'E',)

        # Calculate with dense network forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: deep_out, shape = (B, O = 1)
        deep_out = emb_inputs.flatten(('N', 'E',), 'O')
        deep_out = self.deep(deep_out).sum(dim='O', keepdim=True)

        # Calculate with matrix factorization
        # inputs: emb_inputs, shape = (B, N, E)
        # output: glm_out, shape = (B, O = 1)
        glm_out = self.glm(emb_inputs)

        # Add up glm_out and deep_out
        # inputs: glm_out, shape = (B, O = 1)
        # inputs: deep_out, shape = (B, O = 1)
        # output: output, shape = (B, O = 1)
        outputs = glm_out + deep_out

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
