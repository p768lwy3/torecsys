from . import _CtrModel
from torecsys.layers import GMFLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Callable, List


class NeuralCollaborativeFilteringModel(_CtrModel):
    r"""NeuralCollaborativeFiltering is a model of concatenation of feed forward neural network and 
    generalized matrix factorization, i.e. :math:`y_{mf}(x) = \sum_{v=1}^{k}(x_{u}^{v} * x_{j}^{v})` .

    :Reference:

    #. `Xiangnan He, 2017. Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_.

    """
    @jit_experimental
    def __init__(self, 
                 embed_size       : int,
                 deep_output_size : int,
                 deep_layer_sizes : int,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""inititalize neural collaborative filtering
        
        Args:
            embed_size (int): embedding size
            deep_output_size (int): output size of deep neural network
            deep_layer_sizes (int): layer sizes of deep neural network
            deep_dropout_p (List[float], optional): ropout probability in deep neural network. Allow: [None, list of float for each layer]. Defaults to None.
            deep_activation (Callable[[T], T], optional): activation function of each layer. Allow: [None, Callable[[T], T]]. Defaults to nn.ReLU().
        """
        super(NeuralCollaborativeFilteringModel, self).__init__()
        self.mlp = DNNLayer(
            output_size = deep_output_size, 
            layer_sizes = deep_layer_sizes, 
            embed_size  = embed_size,
            num_fields  = 2,
            dropout_p   = deep_dropout_p, 
            activation  = deep_activation
        )
        self.glm = GMFLayer()

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward calculation of neural network filtering
        
        Args:
            emb_inputs (T), shape = (B, 2, E), dtype = torch.float: features vectors of users and items
        
        Returns:
            T, shape = (B, 1, E), dtype = torch.float: output of neural network filtering
        """
        # mlp_out's input shape = (B, 2, E) and output shape = (B, 1)
        mlp_out = self.mlp(emb_inputs).sum(dim=2)

        # glm_out's input shape = (B, 2, E) and output shape = (B, 1)
        glm_out = self.glm(emb_inputs).squeeze(-1)

        # sum mlp's out and glm's out to model's output
        outputs = glm_out + mlp_out
        
        return outputs
    