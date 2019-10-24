from . import _CtrModel
from torecsys.layers import GMFLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn
from typing import Callable, List

class NeuralCollaborativeFilteringModel(_CtrModel):
    r"""Model class of Neural Collaborative Filtering (NCF).
    
    Neural Collaborative Filtering is a model that concatenate dense network and generalized 
    matrix factorization, i.e. :math:`y_{mf}(x) = \sum_{v=1}^{k}(x_{u}^{v} * x_{j}^{v})`_.

    :Reference:

    #. `Xiangnan He, 2017. Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 embed_size       : int,
                 deep_output_size : int,
                 deep_layer_sizes : int,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize NeuralCollaborativeFilteringModel
        
        Args:
            embed_size (int): Size of embedding tensor
            deep_output_size (int): Output size of dense network
            deep_layer_sizes (int): Layer sizes of dense network
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network. 
                Defaults to nn.ReLU().
        
        Attributes:
            deep (nn.Module): Module of dense layer.
            glm (nn.Module): Module of matrix factorization layer.
        """
        # refer to parent class
        super(NeuralCollaborativeFilteringModel, self).__init__()

        # initialize dense layer
        self.deep = DNNLayer(
            inputs_size = embed_size * 2,
            output_size = deep_output_size, 
            layer_sizes = deep_layer_sizes, 
            dropout_p   = deep_dropout_p, 
            activation  = deep_activation
        )

        # initialize gmf layer
        self.glm = GMFLayer()

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of NeuralCollaborativeFilteringModel
        
        Args:
            emb_inputs (T), shape = (B, N = 2, E), dtype = torch.float: Embedded features tensors of users and items.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of NeuralCollaborativeFilteringModel
        """
        # Calculate with dense network forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: deep_out, shape = (B, O = 1)
        ## deep_out = self.deep(emb_inputs).sum(dim=2)
        deep_out = emb_inputs.unflatten(["N", "E"], "O")
        deep_out = self.deep(deep_out).sum(dim="O", keepdim=True)

        # Calculate with matrix factorization
        # inputs: emb_inputs, shape = (B, N, E)
        # output: glm_out, shape = (B, O = 1)
        glm_out = self.glm(emb_inputs)

        # add up glm_out and deep_out
        # inputs: glm_out, shape = (B, O = 1)
        # inputs: deep_out, shape = (B, O = 1)
        # output: output, shape = (B, O = 1)
        outputs = glm_out + deep_out

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)
        
        return outputs
    