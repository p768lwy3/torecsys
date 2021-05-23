from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import CINLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel


class XDeepFactorizationMachineModel(CtrBaseModel):
    """
    Model class of eXtreme Deep Factorization Machine (xDeepFM).
    
    eXtreme Deep Factorization Machine is a variant of DeepFM by replacing FM part with a 
    Convolutional Neural Network (CNN) based model, called Compress Interaction Network (CIN),
    to calculate element-wise cross-features tensors by outer product, and compress the tensors 
    to 1d by CNN.

    :Reference:

    #. `Jianxun Lian et al, 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
        <https://arxiv.org/abs/1803.05170.pdf>`_.

    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 cin_layer_sizes: List[int],
                 deep_layer_sizes: List[int],
                 cin_is_direct: Optional[bool] = False,
                 cin_use_bias: Optional[bool] = True,
                 cin_use_batchnorm: Optional[bool] = True,
                 cin_activation: Optional[nn.Module] = nn.ReLU(),
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize XDeepFactorizationMachineModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            cin_layer_sizes (List[int]): layer sizes of compress interaction network
            deep_layer_sizes (List[int]): layer sizes of DNN
            cin_is_direct (bool, optional): whether outputs is passed to next step directly or not in compress
                interaction network. Defaults to False
            cin_use_bias (bool, optional): whether bias added to Conv1d or not in compress interaction network.
                Defaults to True
            cin_use_batchnorm (bool, optional): whether batch normalization is applied or not after Conv1d in
                compress interaction network. Defaults to True
            cin_activation (nn.Module, optional): activation function of Conv1d in compress interaction network.
                Allow: [None, Callable[[T], T]]. Defaults to nn.ReLU()
            deep_dropout_p (List[float], optional): probability of Dropout in DNN.
                Allow: [None, list of float for each layer]. Defaults to None
            deep_activation (nn.Module, optional): activation function of Linear. Allow: [None, Callable[[T], T]].
                Defaults to nn.ReLU()
        """
        super().__init__()

        self.cin = CINLayer(
            embed_size=embed_size,
            num_fields=num_fields,
            output_size=1,
            layer_sizes=cin_layer_sizes,
            is_direct=cin_is_direct,
            use_bias=cin_use_bias,
            use_batchnorm=cin_use_batchnorm,
            activation=cin_activation
        )

        self.deep = DNNLayer(
            inputs_size=embed_size * num_fields,
            output_size=1,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of XDeepFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), data_type = torch.float: features tensors
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, 1), data_type = torch.float: output of XDeepFactorizationMachineModel.
        """
        # Name tensors for flatten
        feat_inputs.names = ('B', 'N', 'E',)
        emb_inputs.names = ('B', 'N', 'E',)

        # Reshape inputs of dense layer
        # inputs: feat_inputs, shape = (B, N, E)
        # output: deep_inputs, shape = (B, N * E)
        deep_inputs = emb_inputs.flatten(('N', 'E',), 'E')

        # Forward calculation of cin layer 
        # inputs: emb_inputs, shape = (B, N, E)
        # output: cin_out, shape = (B, O = 1)
        cin_out = self.cin(emb_inputs)

        # Forward calculation of deep layer
        # inputs: deep_inputs, shape = (B, N * E)
        # output: deep_out, shape = (B, O = 1)
        deep_out = self.deep(deep_inputs)

        # Aggregate feat_inputs
        # inputs: feat_inputs, shape = (B, N, 1)
        # output: feat_output, shape = (B, O = 1)
        feat_output = feat_inputs.sum(dim='N')
        feat_output.names = ('B', 'O',)

        # Add up values
        outputs = feat_output + cin_out + deep_out + self.bias

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
