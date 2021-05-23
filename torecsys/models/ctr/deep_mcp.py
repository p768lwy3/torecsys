from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from torecsys.layers import DNNLayer
from torecsys.models.ctr import CtrBaseModel


class DeepMatchingCorrelationPredictionModel(CtrBaseModel):
    r"""
    Model class of Deep Matching, Correlation and Prediction (DeepMCP).
    
    Deep Matching, Correlation and Prediction (DeepMCP) is a model proposed by Wentao Ouyang et al of Alibaba Group in
    2019, which is a model including three parts: Matching, Correlation, and Predation, to adjust the distance between
    user item and item item in the following way:

    #. Prediction subnet: Feed-forward Dense Layers

    #. Matching subnet: sigmoid of dot-product between high-level representations of users and
    items,with the following calculation: :math:`\^{y} = \frac{1}{1 + \text{exp}(-(w^{T}z + b))}`

    #. Correlation subnet: a subnet to control similarity between items 

    :Reference:

    #. `Wentao Ouyang et al, 2019. Representation Learning-Assisted Click-Through Rate Prediction
    <https://arxiv.org/pdf/1906.04365.pdf>`_.

    """

    def __init__(self,
                 embed_size: int,
                 user_num_fields: int,
                 item_num_fields: int,
                 corr_output_size: int,
                 match_output_size: int,
                 corr_layer_sizes: List[int],
                 match_layer_sizes: List[int],
                 pred_layer_sizes: List[int],
                 corr_dropout_p: Optional[List[float]] = None,
                 match_dropout_p: Optional[List[float]] = None,
                 pred_dropout_p: Optional[List[float]] = None,
                 corr_activation: Optional[nn.Module] = nn.ReLU(),
                 match_activation: Optional[nn.Module] = nn.ReLU(),
                 pred_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize DeepMatchingCorrelationPredictionModel
        
        Args:
            embed_size (int): size of embedding tensor
            user_num_fields (int): number of user inputs' fields
            item_num_fields (int): number of item inputs' fields
            corr_output_size (int): output size MLP of correlation subnet
            match_output_size (int): output size MLP of matching subnet
            corr_layer_sizes (List[int]): layer sizes of MLP of correlation subnet
            match_layer_sizes (List[int]): layer sizes of MLP of matching subnet
            pred_layer_sizes (List[int]): layer sizes of MLP of prediction subnet
            corr_dropout_p (List[float], optional): probability of Dropout in MLP of correlation subnet.
                Defaults to None
            match_dropout_p (List[float], optional): probability of Dropout in MLP of matching subnet.
                Defaults to None
            pred_dropout_p (List[float], optional): probability of Dropout in MLP of prediction subnet.
                Defaults to None
            corr_activation (torch.nn.Module, optional): activation function in MLP of correlation subnet.
                Defaults to nn.ReLU()
            match_activation (torch.nn.Module, optional): activation function in MLP of matching subnet.
                Defaults to nn.ReLU()
            pred_activation (torch.nn.Module, optional): activation function in MLP of prediction subnet.
                Defaults to nn.ReLU()
        """
        super().__init__()

        self.prediction = DNNLayer(
            inputs_size=(user_num_fields + item_num_fields) * embed_size,
            output_size=1,
            layer_sizes=pred_layer_sizes,
            dropout_p=pred_dropout_p,
            activation=pred_activation
        )

        self.matching = nn.ModuleDict()
        self.matching['sigmoid'] = nn.Sigmoid()
        self.matching['user'] = nn.Sequential()
        self.matching['user'].add_module('MatchingUserDeep', DNNLayer(
            inputs_size=user_num_fields * embed_size,
            output_size=match_output_size,
            layer_sizes=match_layer_sizes,
            dropout_p=match_dropout_p,
            activation=match_activation
        ))
        self.matching['user'].add_module('MatchingUserActivation', nn.Tanh())
        self.matching['item'] = nn.Sequential()
        self.matching['item'].add_module('MatchingItemDeep', DNNLayer(
            inputs_size=item_num_fields * embed_size,
            output_size=match_output_size,
            layer_sizes=match_layer_sizes,
            dropout_p=match_dropout_p,
            activation=match_activation
        ))
        self.matching['item'].add_module('MatchingItemActivation', nn.Tanh())

        self.correlation = nn.Sequential()
        self.correlation.add_module('CorrelationDeep', DNNLayer(
            inputs_size=item_num_fields * embed_size,
            output_size=corr_output_size,
            layer_sizes=corr_layer_sizes,
            dropout_p=corr_dropout_p,
            activation=corr_activation
        ))
        self.correlation.add_module('CorrelationActivation', nn.Tanh())

    def forward(self,
                user_emb_inputs: torch.Tensor,
                content_emb_inputs: torch.Tensor,
                pos_emb_inputs: torch.Tensor,
                neg_emb_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward calculation of DeepMatchingCorrelationPredictionModel
        
        Args:
            user_emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors of users
            content_emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors of
                content items
            pos_emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors of
                positive sampled items
            neg_emb_inputs (T), shape = (B, N = N Neg, E), data_type = torch.float: embedded features tensors of
                negative sampled items
        
        Returns:
            Tuple[T], shape = (B, O), data_type = torch.float: tuple of output of DeepMCP,
                including prediction scores, matching scores, correlation scores of positive 
                items and negative items.
        """
        # Name inputs tensor for flatten
        user_emb_inputs.names = ('B', 'N', 'E',)
        content_emb_inputs.names = ('B', 'N', 'E',)
        pos_emb_inputs.names = ('B', 'N', 'E',)
        neg_emb_inputs.names = ('B', 'N', 'E',)

        if user_emb_inputs.dim() == 2:
            user_emb_inputs = user_emb_inputs.unflatten('E', (('N', 1,), ('E', user_emb_inputs.size('E'),),))
        elif user_emb_inputs.dim() > 3:
            raise ValueError('Dimension of user_emb_inputs can only be 2 or 3')

        if content_emb_inputs.dim() == 2:
            content_emb_inputs = content_emb_inputs.unflatten('E', (('N', 1,), ('E', content_emb_inputs.size('E'),),))
        elif content_emb_inputs.dim() > 3:
            raise ValueError('Dimension of content_emb_inputs can only be 2 or 3')

        if pos_emb_inputs.dim() == 2:
            pos_emb_inputs = pos_emb_inputs.unflatten('E', (('N', 1), ('E', pos_emb_inputs.size('E'),),))
        elif pos_emb_inputs.dim() > 3:
            raise ValueError('Dimension of pos_emb_inputs can only be 2 or 3')

        if neg_emb_inputs.dim() == 2:
            neg_emb_inputs = neg_emb_inputs.unflatten('E', (('N', 1,), ('E', neg_emb_inputs.size('E'),),))
        elif neg_emb_inputs.dim() > 3:
            raise ValueError('Dimension of neg_emb_inputs can only be 2 or 3')

        # Calculate prediction of prediction subnet
        # Concatenate user and content on dimension of N, which return shape = (B, N, E)
        # Then flatten to shape = (B, N * E)
        # Return y_pred with shape = (B, O = 1)
        cat_pred = torch.cat([user_emb_inputs, content_emb_inputs], dim='N')
        cat_pred = cat_pred.flatten(('N', 'E',), 'E')
        y_pred = self.prediction(cat_pred)

        # Calculate inference of matching subnet,
        # Return (user_match, item_match) with shape = (B, O)
        # Then multiply them and sum on dimension O, which return shape = (B, O)
        # Finally apply sigmoid to y_pred
        user_match = self.matching['user'](user_emb_inputs.flatten(('N', 'E',), 'E'))
        item_match = self.matching['item'](content_emb_inputs.flatten(('N', 'E',), 'E'))
        y_match = (user_match * item_match).sum(dim='O', keepdim=True)
        y_match = self.matching['sigmoid'](y_match)

        # Calculate features' representations of items and return shape = (B, 1 or N, O_C)
        content_corr = self.correlation(content_emb_inputs.flatten(('N', 'E',), 'E'))
        positive_corr = self.correlation(pos_emb_inputs.flatten(('N', 'E',), 'E'))
        negative_corr = self.correlation(neg_emb_inputs.flatten(('N', 'E',), 'E'))

        # Calculate the inference of correlation subnet between content and positive or negative by dot-product,
        # Return a pair of tensors with shape = (B, 1)
        y_corr_pos = (content_corr * positive_corr).sum(dim='O')
        y_corr_neg = (content_corr * negative_corr).sum(dim='O')  # .mean(dim='N', keepdim=True)

        # Since auto grad does not support Named Tensor at this stage, drop the name of output tensor.
        y_pred = y_pred.rename(None)
        y_match = y_match.rename(None)
        y_corr_pos = y_corr_pos.rename(None).unsqueeze(-1)
        y_corr_neg = y_corr_neg.rename(None).unsqueeze(-1)

        return y_pred, y_match, y_corr_pos, y_corr_neg
