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
        # Refer to parent class
        super(DeepFactorizationMachineModel, self).__init__()

        # Initialize fm layer
        self.fm = FMLayer(fm_dropout_p)

        # Initialize dense layer
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
        # Reshape feat_inputs 
        # inputs: feat_inputs, shape = (B, N, 1) 
        # output: fm_first, shape = (B, O = N)
        fm_first = feat_inputs.flatten(["N", "E"], "O")

        # Calculate with fm layer forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: fm_second, shape = (B, O = E)
        fm_second = self.fm(emb_inputs)

        # Concatenate fm_second and fm_first on dimension O
        # inputs: fm_second, shape = (B, O = E)
        # inputs: fm_first, shape = (B, O = N)
        # output: fm_out, shape = (B, O = E + N)
        fm_out = torch.cat([fm_second, fm_first], dim="O")

        # Aggregate fm_out on dimension O
        # inputs: fm_out, shape = (B, O)
        # output: fm_out, shape = (B, O = 1)
        fm_out = fm_out.sum(dim="O", keepdim=True)

        # Calculate with dense layer forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: deep_out, shape = (B, O = 1)
        deep_out = self.deep(emb_inputs)

        # Add up deep_out and fm_out
        # inputs: deep_out, shape = (B, O = 1)
        # inputs: fm_out, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = deep_out + fm_out
        
        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
