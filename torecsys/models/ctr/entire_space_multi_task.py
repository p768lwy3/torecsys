from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from torecsys.layers import DNNLayer
from torecsys.models.ctr import CtrBaseModel


class EntireSpaceMultiTaskModel(CtrBaseModel):
    """
    Model class of Entire Space Multi Task Model (ESMM)

    Entire Space Multi Task Model is a model applying transfer learning on recommendation system in a straightforward
    way, which is a pair of pooling and dense networks to calculate prediction of click through rate (CTR) and
    conversion rate (CVR) sharing a set of embedding tensors, and compute losses with CTR and a multiple of CTR and
    CVR, called CTCVR.

    :Reference:

    #. `Xiao Ma et al, 2018. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion
        Rate <https://arxiv.org/abs/1804.07931>`_.

    """

    def __init__(self,
                 num_fields: int,
                 layer_sizes: List[int],
                 dropout_p: Optional[List[float]] = None,
                 activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize EntireSpaceMultiTaskModel
        
        Args:
            num_fields (int): number of inputs' fields
            layer_sizes (List[int]): layer sizes of dense network
            dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None.
            activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU().
        """
        super().__init__()

        self.cvr_pooling = nn.AdaptiveAvgPool1d(1)
        self.ctr_pooling = nn.AdaptiveAvgPool1d(1)
        self.cvr_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)
        self.ctr_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)

    def forward(self, emb_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward calculation of EntireSpaceMultiTaskModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors.
        
        Returns:
            Tuple[T], shape = (B, O), data_type = torch.float: tuple of output of EntireSpaceMultiTaskModel,
                including probability of CVR prediction (pcvr) and probability of CTR prediction (pctr).
        """
        # Pool inputs for CVR prediction and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_cvr, shape = (B, N)
        pooled_cvr = self.cvr_pooling(emb_inputs.rename(None))
        pooled_cvr.names = ('B', 'N', 'E',)
        pooled_cvr = pooled_cvr.flatten(('N', 'E',), 'N')

        # Calculate with dense layer of CVR prediction
        # inputs: pooled_cvr, shape = (B, N)
        # output: pcvr, shape = (B, 1)
        pcvr = self.cvr_deep(pooled_cvr)

        # Pool inputs for CTR prediction and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_ctr, shape = (B, N)
        pooled_ctr = self.ctr_pooling(emb_inputs.rename(None))
        pooled_ctr.names = ('B', 'N', 'E',)
        pooled_ctr = pooled_ctr.flatten(('N', 'E',), 'N')

        # Calculate with dense layer of CTR prediction
        # inputs: pooled_ctr, shape = (B, N)
        # output: pctr, shape = (B, 1)
        pctr = self.ctr_deep(pooled_ctr)

        return pcvr, pctr
