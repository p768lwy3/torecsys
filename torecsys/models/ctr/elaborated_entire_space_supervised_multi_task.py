from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from torecsys.layers import DNNLayer
from torecsys.models.ctr import CtrBaseModel


class ElaboratedEntireSpaceSupervisedMultiTaskModel(CtrBaseModel):
    """
    Model class of Elaborated Entire Space Supervised Multi Task Model (ESM2).

    Elaborated Entire Space Supervised Multi Task Model is a variant of Entire Space Multi Task Model, which is to
    handle missed actions to order, like cart, wish, like etc, by adding two more base model to predict the direct
    CVR (Deterministic Action) and non-direct CVR separately (Other Action).

    :Reference:

    #. `Hong Wen et al, 2019. Conversion Rate Prediction via Post-Click Behaviour Modeling
    <https://arxiv.org/abs/1910.07099>`_.

    """

    def __init__(self,
                 num_fields: int,
                 layer_sizes: List[int],
                 dropout_p: Optional[List[float]] = None,
                 activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize ElaboratedEntireSpaceSupervisedMultiTaskModel
        
        Args:
            num_fields (int): number of inputs' fields
            layer_sizes (List[int]): layer sizes of dense network
            dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            activation (Callable[[T], T], optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.impress_to_click_pooling = nn.AdaptiveAvgPool1d(1)
        self.click_to_d_action_pooling = nn.AdaptiveAvgPool1d(1)
        self.d_action_to_buy_pooling = nn.AdaptiveAvgPool1d(1)
        self.o_action_to_buy_pooling = nn.AdaptiveAvgPool1d(1)

        self.impress_to_click_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)
        self.click_to_d_action_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)
        self.d_action_to_buy_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)
        self.o_action_to_buy_deep = DNNLayer(num_fields, 1, layer_sizes, dropout_p, activation)

    def forward(self, emb_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward calculation of ElaboratedEntireSpaceSupervisedMultiTaskModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            Tuple[T], shape = (B, O), data_type = torch.float: tuple of output of
                ElaboratedEntireSpaceSupervisedMultiTaskModel, including probability of impression to click,
                probability of impression to DAction and probability of impression to buy
        """
        # Pool inputs for impress_to_click and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_impress_to_click, shape = (B, N)
        pooled_impress_to_click = self.impress_to_click_pooling(emb_inputs.rename(None))
        pooled_impress_to_click.names = ('B', 'N', 'E',)
        pooled_impress_to_click = pooled_impress_to_click.flatten(('N', 'E',), 'N')

        # Pool inputs for click_to_d_action and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_click_to_d_action, shape = (B, N)
        pooled_click_to_d_action = self.click_to_d_action_pooling(emb_inputs.rename(None))
        pooled_click_to_d_action.names = ('B', 'N', 'E',)
        pooled_click_to_d_action = pooled_click_to_d_action.flatten(('N', 'E',), 'N')

        # Pool inputs for d_action_to_buy and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_d_action_to_buy, shape = (B, N)
        pooled_d_action_to_buy = self.d_action_to_buy_pooling(emb_inputs.rename(None))
        pooled_d_action_to_buy.names = ('B', 'N', 'E',)
        pooled_d_action_to_buy = pooled_d_action_to_buy.flatten(('N', 'E',), 'N')

        # Pool inputs for o_action_to_buy and flatten it
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_o_action_to_buy, shape = (B, N)
        pooled_o_action_to_buy = self.o_action_to_buy_pooling(emb_inputs.rename(None))
        pooled_o_action_to_buy.names = ('B', 'N', 'E',)
        pooled_o_action_to_buy = pooled_o_action_to_buy.flatten(('N', 'E',), 'N')

        # Calculate with dense layer of impress_to_click
        # inputs: pooled_impress_to_click, shape = (B, N)
        # output: prob_impress_to_click, shape = (B, 1)
        prob_impress_to_click = self.impress_to_click_deep(pooled_impress_to_click)

        # Calculate with dense layer of click_to_d_action
        # inputs: pooled_click_to_d_action, shape = (B, N)
        # output: prob_click_to_d_action, shape = (B, 1)
        prob_click_to_d_action = self.click_to_d_action_deep(pooled_click_to_d_action)

        # Calculate with dense layer of d_action_to_buy
        # inputs: pooled_d_action_to_buy, shape = (B, N)
        # output: prob_d_action_to_buy, shape = (B, 1)
        prob_d_action_to_buy = self.d_action_to_buy_deep(pooled_d_action_to_buy)

        # Calculate with dense layer of o_action_to_buy
        # inputs: pooled_o_action_to_buy, shape = (B, N)
        # output: prob_o_action_to_buy, shape = (B, 1)
        prob_o_action_to_buy = self.o_action_to_buy_deep(pooled_o_action_to_buy)

        # Calculate probability from impress to DAction
        # inputs: prob_impress_to_click, shape = (B, 1)
        # inputs: prob_click_to_d_action, shape = (B, 1)
        # output: prob_impress_to_d_action, shape = (B, 1)
        prob_impress_to_d_action = prob_impress_to_click * prob_click_to_d_action

        # Calculate probability from impress to buy
        # output: prob_impress_to_buy, shape = (B, 1)
        prob_click_d_action_buy = prob_click_to_d_action * prob_d_action_to_buy
        prob_click_o_action_buy = (1 - prob_click_d_action_buy) * prob_o_action_to_buy
        prob_click_to_buy = prob_click_d_action_buy + prob_click_o_action_buy
        prob_impress_to_buy = prob_impress_to_click * prob_click_to_buy

        return prob_impress_to_click, prob_impress_to_d_action, prob_impress_to_buy
