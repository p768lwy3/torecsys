from . import _CtrModel
from torecsys.layers import FMLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn
from typing import Callable, List


class DeepFactorizationMachineModel(_CtrModel):
    r"""DeepFactorizationMachineModel is a model of Deep Factorization Machine (DeepFM) proposed
    by Huawei in 2017, which add up outputs of factorization machine and fully-connected dense 
    neural network directly: :math:`y_{out} = y_{deep} + y_{fm}` , to gain the advantage of two 
    different models of two different objectives, i.e. to gain the explainable power in high dimension 
    of Deep Neural Network, and to gain the explainable power in low dimension of Factorization 
    Machine, Hence, the :math:`y_{deep}` and :math:`y_{fm}` are calculated with the following 
    equations:

    #. for the deep part, :math:`y_{deep}` is the outcome of a Deep Feed-forward Neural Network, 
    which is equal to :math:`y_{i} = \text{activation} ( W_{i} a_{i - 1} + b_{i} )` .

    #. and for the fm part, :math:`y_{fm}` is the result of a factorization machine calculate, 
    which is equal to :math:`y_{fm} = \sum_{i=1}^{n} w_{i} x_{i} + \sum_{i=1}^{n} \sum_{j=1+1}^{n} <v_{i},v_{j}> x_{i} x_{j}` .

    :Reference:

    #. `Huifeng Guo et al, 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction <https://arxiv.org/abs/1703.04247>`_.
    """
    @jit_experimental
    def __init__(self, 
                 embed_size       : int,
                 num_fields       : int,
                 deep_layer_sizes : List[int],
                 fm_dropout_p     : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""initialize Deep Factorization Machine Model
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            deep_layer_sizes (List[int]): layer sizes of deep neural network
            fm_dropout_p (float, optional): dropout probability after factorization machine. Defaults to 0.0.
            deep_dropout_p (List[float], optional): dropout probability after activation of each layer. Allow: [None, list of float for each layer]. Defaults to None.
            deep_activation (Callable[[T], T], optional): activation function of each layer. Allow: [None, Callable[[T], T]]. Defaults to nn.ReLU().
        
        Attributes:
            fm (nn.Module): module of FM layer
            deep (nn.Module): module of dense layer
        """
        # initialize nn.Module class
        super(DeepFactorizationMachineModel, self).__init__()

        # layers (deep and fm) of second-order part of inputs
        self.fm = FMLayer(fm_dropout_p)

        self.deep = DNNLayer(
            output_size = 1,
            layer_sizes = deep_layer_sizes,
            embed_size  = embed_size,
            num_fields  = num_fields,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        )
    
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of Deep Factorization Machine Model
        
        Args:
            feat_inputs (T), shape = (B, N, 1): first order outputs, i.e. outputs from nn.Embedding(V, 1)
            emb_inputs (T), shape = (B, N, E): second order outputs of one-hot encoding, i.e. outputs from nn.Embedding(V, E)
        
        Returns:
            T, shape = (B, O), dtype = torch.float: outputs of Deep Factorization Machine Model
        """

        # feat_inputs'shape = (B, N, 1) and reshape to (B, N)
        ## fm_first = feat_inputs.squeeze()
        if feat_inputs.dim() == 2:
            fm_first = feat_inputs
            fm_first.names = ("B", "O")
        elif feat_inputs.dim() == 3:
            fm_first = feat_inputs.flatten(["N", "E"], "O")
        else:
            raise ValueError("Dimension of feat_inputs can only be 2 or 3.")

        # pass to fm layer where its returns' shape = (B, O)
        fm_second = self.fm(emb_inputs)

        # calculate output of factorization machine with output's shape = (B, O = 1)
        fm_out = torch.cat([fm_first, fm_second], dim="O")
        fm_out = fm_out.sum(dim="O", keepdim=True)

        # pass to dense layers with output's shape = (B, O = 1)
        deep_out = self.deep(emb_inputs)

        # deepfm outputs = fm_out + deep_out
        outputs = deep_out + fm_out
        
        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        outputs = outputs.rename(None)

        return outputs
