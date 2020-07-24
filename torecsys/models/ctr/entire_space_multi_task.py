from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from torecsys.layers import DNNLayer
from torecsys.utils.decorator import no_jit_experimental_by_namedtensor
from . import _CtrModel


class EntireSpaceMultiTaskModel(_CtrModel):
    r"""Model class of Entire Space Multi Task Model (ESMM).

    Entire Space Multi Task Model is a model applying transfer learning on recommendation 
    system in a straightforward way, which is a pair of pooling and dense networks to calculate 
    prediction of click through rate (CTR) and conversion rate (CVR) sharing a set of embedding 
    tensors, and compute losses with CTR and a multipe of CTR and CVR, called CTCVR.

    :Reference:

    #. `Xiao Ma et al, 2018. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate <https://arxiv.org/abs/1804.07931>`_.

    """

    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 num_fields: int,
                 layer_sizes: List[int],
                 dropout_p: List[float] = None,
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize EntireSpaceMultiTaskModel
        
        Args:
            num_fields (int): Number of inputs' fields
            layer_sizes (List[int]): Layer sizes of dense network
            dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            activation (Callable[[T], T], optional): Activation function of dense network. 
                Defaults to nn.ReLU().
        
        Attributes:
            cvr_pooling (nn.Module): Module of 1D average pooling layer for CVR prediction.
            cvr_deep (nn.Module): Module of dense layer.
            ctr_pooling (nn.Module): Module of 1D average pooling layer for CTR prediction.
            ctr_deep (nn.Module): Module of dense layer.
        """
        # Refer to parent class
        super(EntireSpaceMultiTaskModel, self).__init__()

        # Initiailze pooling layer of CVR
        self.cvr_pooling = nn.AdaptiveAvgPool1d(1)

        # Initialize dense layer of CVR
        self.cvr_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)

        # Initialize pooling layer of CTR
        self.ctr_pooling = nn.AdaptiveAvgPool1d(1)

        # Initialize dense layer of CTR
        self.ctr_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)

    def forward(self, emb_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        r"""Forward calculation of EntireSpaceMultiTaskModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            Tuple[T], shape = (B, O), dtype = torch.float: Tuple of output of EntireSpaceMultiTaskModel,
                including probability of CVR prediction (pcvr) and probability of CTR prediction (pctr).
        """
        # Pool inputs for CVR prediction and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_cvr, shape = (B, N)
        pooled_cvr = self.cvr_pooling(emb_inputs.rename(None))
        pooled_cvr.names = ("B", "N", "E")
        pooled_cvr = pooled_cvr.flatten(["N", "E"], "N")

        # Calculate with dense layer of CVR prediction
        # inputs: pooled_cvr, shape = (B, N)
        # output: pcvr, shape = (B, 1)
        pcvr = self.cvr_deep(pooled_cvr)

        # Pool inputs for CTR prediction and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_ctr, shape = (B, N)
        pooled_ctr = self.ctr_pooling(emb_inputs.rename(None))
        pooled_ctr.names = ("B", "N", "E")
        pooled_ctr = pooled_ctr.flatten(["N", "E"], "N")

        # Calculate with dense layer of CTR prediction
        # inputs: pooled_ctr, shape = (B, N)
        # output: pctr, shape = (B, 1)
        pctr = self.ctr_deep(pooled_ctr)

        return pcvr, pctr
