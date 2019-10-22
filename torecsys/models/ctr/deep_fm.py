from . import _CtrModel
from torecsys.layers import FMLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn
from typing import Callable, List


class DeepFactorizationMachineModel(_CtrModel):
    r"""Model class of Deep Factorization Machine (Deep FM).
    
    Deep Factorization Machine is a model proposed by Huawei in 2017, which sum up outputs 
    of factorization machine and fully-connected dense network directly, to gain the advantages 
    from two different models of two different objectives, i.e. to gain the explainable power 
    in high dimension of dense network, and to gain the explainable power in low dimension of 
    factorization machine.

    :Reference:

    #. `Huifeng Guo et al, 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction <https://arxiv.org/abs/1703.04247>`_.
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 embed_size       : int,
                 num_fields       : int,
                 deep_layer_sizes : List[int],
                 fm_dropout_p     : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize DeepFactorizationMachineModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            deep_layer_sizes (List[int]): Layer sizes of dense network
            fm_dropout_p (float, optional): Probability of Dropout in FM. 
                Defaults to 0.0.
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network. 
                Defaults to nn.ReLU().
        
        Attributes:
            fm (nn.Module): Module of factorization machine layer.
            deep (nn.Module): Module of dense layer.
        """
        # refer to parent class
        super(DeepFactorizationMachineModel, self).__init__()

        # initialize fm layer
        self.fm = FMLayer(fm_dropout_p)

        # initialize dense layer
        self.deep = DNNLayer(
            output_size = 1,
            layer_sizes = deep_layer_sizes,
            embed_size  = embed_size,
            num_fields  = num_fields,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        )
    
    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of DeepFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: Linear Features tensors.
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of DeepFactorizationMachineModel.
        """
        ## fm_first = feat_inputs.squeeze()
        if feat_inputs.dim() == 2:
            fm_first = feat_inputs
            fm_first.names = ("B", "O")
        elif feat_inputs.dim() == 3:
            # reshape feat_inputs from (B, N, 1) to (B, N)
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
