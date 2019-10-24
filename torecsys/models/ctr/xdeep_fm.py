from . import _CtrModel
from torecsys.layers import CINLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Callable, List


class xDeepFactorizationMachineModel(_CtrModel):
    r"""Model class of eXtreme Deep Factorization Machine (xDeepFM).
    
    eXtreme Deep Factorization Machine is a variant of DeepFM by replacing FM part with a 
    Convalutional Neural Network (CNN) based model, called Compress Interaction Network (CIN), 
    to calculate element-wise cross-features tensors by outer product, and compress the tensors 
    to 1d by CNN.

    :Reference:

    #. `Jianxun Lian et al, 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems <https://arxiv.org/abs/1803.05170.pdf>`_.

    """
    def __init__(self, 
                 embed_size        : int,
                 num_fields        : int,
                 cin_layer_sizes   : List[int],
                 deep_layer_sizes  : List[int],
                 cin_is_direct     : bool = False,
                 cin_use_bias      : bool = True,
                 cin_use_batchnorm : bool = True,
                 cin_activation    : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 deep_dropout_p    : List[float] = None,
                 deep_activation   : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize xDeepFactorizationMachineModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            cin_layer_sizes (List[int]): Layer sizes of compress interaction network
            deep_layer_sizes (List[int]): Layer sizes of DNN
            cin_is_direct (bool, optional): Whether outputs is passed to next step directly or not in compress interaction network.
                Defaults to False.
            cin_use_bias (bool, optional): Whether bias added to Conv1d or not in compress interaction network. 
                Defaults to True.
            cin_use_batchnorm (bool, optional): Whether batch normalization is applied or not after Conv1d in compress interaction network. 
                Defaults to True.
            cin_activation (Callable[[T], T], optional): Activation function of Conv1d in compress interaction network. 
                Allow: [None, Callable[[T], T]]. 
                Defaults to nn.ReLU().
            deep_dropout_p (List[float], optional): Probability of Dropout in DNN. 
                Allow: [None, list of float for each layer]. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of Linear. 
                Allow: [None, Callable[[T], T]]. 
                Defaults to nn.ReLU().
        
        Attributes:
            cin (torecsys.layers.CompressInteractionNetworkLayer): Compress Interaction Network Layer.
            deep (torecsys.layers.MultilayerPerceptronLayer): Deep Neural Network Layer.
            bias (torch.nn.Tensor): Bias variable, which is a trainable tensor.
        """
        # refer to parent class
        super(xDeepFactorizationMachineModel, self).__init__()

        # initialize cin layer
        self.cin = CINLayer(
            embed_size = embed_size,
            num_fields = num_fields,
            output_size = 1,
            layer_sizes = cin_layer_sizes,
            is_direct = cin_is_direct,
            use_bias = cin_use_bias,
            use_batchnorm = cin_use_batchnorm,
            activation = cin_activation
        )

        # initialize deep layer
        self.deep = DNNLayer(
            output_size=1,
            layer_sizes=deep_layer_sizes,
            embed_size=embed_size,
            num_fields=num_fields,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

        # initialize bias variable
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.bias.data)
    
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of xDeepFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: Features tensors.
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: Output of xDeepFactorizationMachineModel.
        """

        # forward calculation of cin layer and deep layer
        cin_out = self.cin(emb_inputs).squeeze().sum(dim=1, keepdim=True)
        deep_out = self.deep(emb_inputs).squeeze().sum(dim=1, keepdim=True)

        # sum all values as outputs
        outputs = feat_inputs.squeeze().sum(dim=1, keepdim=True)
        outputs = outputs + cin_out + deep_out + self.bias

        return outputs
