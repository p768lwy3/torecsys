from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from torecsys.layers import DNNLayer
from torecsys.utils.decorator import no_jit_experimental_by_namedtensor
from . import _CtrModel


class ElaboratedEntireSpaceSupervisedMultiTaskModel(_CtrModel):
    r"""Model class of Elaborated Entire Space Supervised Multi Task Model (ESM2).

    Elaborated Entire Space Supervised Multi Task Model is a variant of Entire Space 
    Multi Task Model, which is to handle missed actions to order, like cart, wish, like 
    etc, by adding two more base model to predict the direct CVR (Deterministic Action) 
    and non-direct CVR separately (Other Action).

    :Reference:

    #. `Hong Wen et al, 2019. Conversion Rate Prediction via Post-Click Behaviour Modeling
    <https://arxiv.org/abs/1910.07099>`_.

    """

    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 num_fields: int,
                 layer_sizes: List[int],
                 dropout_p: List[float] = None,
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize ElaboratedEntireSpaceSupervisedMultiTaskModel
        
        Args:
            num_fields (int): Number of inputs' fields
            layer_sizes (List[int]): Layer sizes of dense network
            dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            activation (Callable[[T], T], optional): Activation function of dense network. 
                Defaults to nn.ReLU().
        
        Attributes:
            impress_to_click_pooling (nn.Module): Module of 1D average pooling layer for impress_to_click
            click_to_daction_pooling (nn.Module): Module of 1D average pooling layer for click_to_daction
            daction_to_buy_pooling (nn.Module): Module of 1D average pooling layer for daction_to_buy
            oaction_to_buy_pooling (nn.Module): Module of 1D average pooling layer for oaction_to_buy
            impress_to_click_deep (nn.Module): Module of dense layer.
            click_to_daction_deep (nn.Module): Module of dense layer.
            daction_to_buy_deep (nn.Module): Module of dense layer.
            oaction_to_buy_deep (nn.Module): Module of dense layer.
        """
        # Refer to parent class
        super(ElaboratedEntireSpaceSupervisedMultiTaskModel, self).__init__()

        # Initialize pooling layers
        self.impress_to_click_pooling = nn.AdaptiveAvgPool1d(1)
        self.click_to_daction_pooling = nn.AdaptiveAvgPool1d(1)
        self.daction_to_buy_pooling = nn.AdaptiveAvgPool1d(1)
        self.oaction_to_buy_pooling = nn.AdaptiveAvgPool1d(1)

        # Initialize dense layers
        self.impress_to_click_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)
        self.click_to_daction_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)
        self.daction_to_buy_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)
        self.oaction_to_buy_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)

    def forward(self, emb_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        r"""Forward calculation of ElaboratedEntireSpaceSupervisedMultiTaskModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            Tuple[T], shape = (B, O), dtype = torch.float: Tuple of output of ElaboratedEntireSpaceSupervisedMultiTaskModel,
                including probability of impression to click, probability of impression to DAction and probability of
                impression to buy.
        """
        # Pool inputs for impress_to_click and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_impress_to_click, shape = (B, N)
        pooled_impress_to_click = self.impress_to_click_pooling(emb_inputs.rename(None))
        pooled_impress_to_click.names = ("B", "N", "E")
        pooled_impress_to_click = pooled_impress_to_click.flatten(["N", "E"], "N")

        # Pool inputs for click_to_daction and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_click_to_daction, shape = (B, N)
        pooled_click_to_daction = self.click_to_daction_pooling(emb_inputs.rename(None))
        pooled_click_to_daction.names = ("B", "N", "E")
        pooled_click_to_daction = pooled_click_to_daction.flatten(["N", "E"], "N")

        # Pool inputs for daction_to_buy and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_daction_to_buy, shape = (B, N)
        pooled_daction_to_buy = self.daction_to_buy_pooling(emb_inputs.rename(None))
        pooled_daction_to_buy.names = ("B", "N", "E")
        pooled_daction_to_buy = pooled_daction_to_buy.flatten(["N", "E"], "N")

        # Pool inputs for oaction_to_buy and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_oaction_to_buy, shape = (B, N)
        pooled_oaction_to_buy = self.oaction_to_buy_pooling(emb_inputs.rename(None))
        pooled_oaction_to_buy.names = ("B", "N", "E")
        pooled_oaction_to_buy = pooled_oaction_to_buy.flatten(["N", "E"], "N")

        # Calculate with dense layer of impress_to_click
        # inputs: pooled_impress_to_click, shape = (B, N)
        # output: prob_impress_to_click, shape = (B, 1)
        prob_impress_to_click = self.impress_to_click_deep(pooled_impress_to_click)

        # Calculate with dense layer of click_to_daction
        # inputs: pooled_click_to_daction, shape = (B, N)
        # output: prob_click_to_daction, shape = (B, 1)
        prob_click_to_daction = self.click_to_daction_deep(pooled_click_to_daction)

        # Calculate with dense layer of daction_to_buy
        # inputs: pooled_daction_to_buy, shape = (B, N)
        # output: prob_daction_to_buy, shape = (B, 1)
        prob_daction_to_buy = self.daction_to_buy_deep(pooled_daction_to_buy)

        # Calculate with dense layer of oaction_to_buy
        # inputs: pooled_oaction_to_buy, shape = (B, N)
        # output: prob_oaction_to_buy, shape = (B, 1)
        prob_oaction_to_buy = self.oaction_to_buy_deep(pooled_oaction_to_buy)

        # Calculate probability from impress to DAction
        # inputs: prob_impress_to_click, shape = (B, 1)
        # inputs: prob_click_to_daction, shape = (B, 1)
        # output: prob_impress_to_daction, shape = (B, 1)
        prob_impress_to_daction = prob_impress_to_click * prob_click_to_daction

        # Calculate probability from impress to buy
        # inputs: prob_impress_to_click, shape = (B, 1)
        # inputs: prob_click_to_daction, shape = (B, 1)
        # inputs: prob_click_to_daction, shape = (B, 1)
        # inputs: prob_click_to_daction, shape = (B, 1)
        # output: prob_impress_to_buy, shape = (B, 1)
        prob_click_daction_buy = prob_click_to_daction * prob_daction_to_buy
        prob_click_oaction_buy = (1 - prob_click_daction_buy) * prob_oaction_to_buy
        prob_click_to_buy = prob_click_daction_buy + prob_click_oaction_buy
        prob_impress_to_buy = prob_impress_to_click * prob_click_to_buy

        return prob_impress_to_click, prob_impress_to_daction, prob_impress_to_buy
