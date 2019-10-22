from . import _CtrModel
from torecsys.layers import DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn
from typing import Callable, List, Tuple


class DeepMatchingCorrelationPredictionModel(_CtrModel):
    r"""Model class of Deep Matching, Correlation and Prediction (DeepMCP).
    
    Deep Matching, Correlation and Preidction (DeepMCP) is a model proposed by Wentao Ouyang 
    et al of Alibaba Group in 2019, which is a model including three parts: Matching, 
    Correlation, and Predction, to adjust the distance between user item and item item in the 
    following way: 

    #. Prediction subnet: Feed-forward Dense Layers

    #. Matching subnt: sigmoid of dot-product between high-level representations of users and 
    items,with the following calculation: :math"`\^{y} = \frac{1}{1 + \text{exp}(-(w^{T}z + b))}`_.

    #. Correlation subnet: a subnet to control similarity between items 

    :Reference:

    #. `Wentao Ouyang et al, 2019. Representation Learning-Assisted Click-Through Rate Prediction <https://arxiv.org/pdf/1906.04365.pdf>`

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size        : int,
                 user_num_fields   : int,
                 item_num_fields   : int,
                 corr_output_size  : int,
                 match_output_size : int,
                 corr_layer_sizes  : List[int],
                 match_layer_sizes : List[int],
                 pred_layer_sizes  : List[int],
                 corr_dropout_p    : List[float] = None,
                 match_dropout_p   : List[float] = None,
                 pred_dropout_p    : List[float] = None,
                 corr_activation   : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 match_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 pred_activation   : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize DeepMatchingCorrelationPredictionModel
        
        Args:
            embed_size (int): Size of embedding tensor
            user_num_fields (int): Number of user inputs' fields
            item_num_fields (int): Number of item inputs' fields
            corr_output_size (int): Output size MLP of correlation subnet
            match_output_size (int): Output size MLP of matching subnet
            corr_layer_sizes (List[int]): Layer sizes of MLP of correlation subnet
            match_layer_sizes (List[int]): Layer sizes of MLP of matching subnet
            pred_layer_sizes (List[int]): Layer sizes of MLP of prediction subnet
            corr_dropout_p (List[float], optional): Probability of Dropout in MLP of correlation subnet. 
                Defaults to None.
            match_dropout_p (List[float], optional): Probability of Dropout in MLP of matching subnet. 
                Defaults to None.
            pred_dropout_p (List[float], optional): Probability of Dropout in MLP of prediction subnet. 
                Defaults to None.
            corr_activation (Callable[[T], T], optional): Activation function in MLP of correlation subnet. 
                Defaults to nn.ReLU().
            match_activation (Callable[[T], T], optional): Activation function in MLP of matching subnet.
                Defaults to nn.ReLU().
            pred_activation (Callable[[T], T], optional): Activation function in MLP of prediction subnet.
                Defaults to nn.ReLU().
        """
        # refer to parent class
        super(DeepMatchingCorrelationPredictionModel, self).__init__()

        # initialize prediction subnet, which is a dense network with output's size = 1
        self.prediction = DNNLayer(
            inputs_size = (user_num_fields + item_num_fields) * embed_size,
            output_size = 1,
            layer_sizes = pred_layer_sizes,
            dropout_p   = pred_dropout_p,
            activation  = pred_activation
        )

        # initialize matching subnet, which is a pair of dense networks to calculate high-level 
        # representations of users and items, and calculate sigmoid of dot product between them.
        self.matching = nn.ModuleDict()
        self.matching["sigmoid"] = nn.Sigmoid()

        # initialize user part of matching subnet
        self.matching["user"] = nn.Sequential()
        self.matching["user"].add_module(
            "deep", DNNLayer(
                inputs_size = user_num_fields * embed_size,
                output_size = match_output_size,
                layer_sizes = match_layer_sizes,
                dropout_p   = match_dropout_p,
                activation  = match_activation
            )
        )
        self.matching["user"].add_module("activation", nn.Tanh())
        
        # initialize item part of matching subnet 
        self.matching["item"] = nn.Sequential()
        self.matching["item"].add_module(
            "deep", DNNLayer(
                inputs_size = item_num_fields * embed_size,
                output_size = match_output_size,
                layer_sizes = match_layer_sizes,
                dropout_p   = match_dropout_p,
                activation  = match_activation
            )
        )
        self.matching["item"].add_module("activation", nn.Tanh())

        # initialize correlation subnet
        self.correlation = nn.Sequential()
        self.correlation.add_module(
            "deep", DNNLayer(
                inputs_size = item_num_fields * embed_size,
                output_size = corr_output_size,
                layer_sizes = corr_layer_sizes,
                dropout_p   = corr_dropout_p,
                activation  = corr_activation
            )
        )
        self.correlation.add_module("activation", nn.Tanh())

    def forward(self, 
                user_emb_inputs    : torch.Tensor, 
                content_emb_inputs : torch.Tensor, 
                pos_emb_inputs     : torch.Tensor, 
                neg_emb_inputs     : torch.Tensor) -> Tuple[torch.Tensor]:
        r"""Forward calculation of DeepMatchingCorrelationPredictionModel
        
        Args:
            user_emb_inputs (T), shape = (B, N = 1, E), dtype = torch.float: Embedded features tensors of users.
            content_emb_inputs (T), shape = (B, N = 1, E), dtype = torch.float: Embedded features tensors of content items.
            pos_emb_inputs (T), shape = (B, N = 1, E), dtype = torch.float: Embedded features tensors of positive sampled items.
            neg_emb_inputs (T), shape = (B, N = Nneg, E), dtype = torch.float: Embedded features tensors of negative sampled items.
        
        Returns:
            Tuple[T], shape = (B, O), dtype = torch.float: Tuple of output of DeepMCP, 
                including prediction scores, matching scores, correlation scores of positive 
                items and negative items.
        """
        # check if the inputs' dimension are correct
        if user_emb_inputs.dim() == 2:
            user_emb_inputs = user_emb_inputs.unflatten("E", [("N", 1), ("E", user_emb_inputs.size("E"))])
        elif user_emb_inputs.dim() > 3:
            raise ValueError("Dimension of user_emb_inputs can only be 2 or 3.")
        
        if content_emb_inputs.dim() == 2:
            content_emb_inputs = content_emb_inputs.unflatten("E", [("N", 1), ("E", content_emb_inputs.size("E"))])
        elif content_emb_inputs.dim() > 3:
            raise ValueError("Dimension of content_emb_inputs can only be 2 or 3.")

        if pos_emb_inputs.dim() == 2:
            pos_emb_inputs = pos_emb_inputs.unflatten("E", [("N", 1), ("E", pos_emb_inputs.size("E"))])
        elif pos_emb_inputs.dim() > 3:
            raise ValueError("Dimension of pos_emb_inputs can only be 2 or 3.")
        
        if neg_emb_inputs.dim() == 2:
            neg_emb_inputs = neg_emb_inputs.unflatten("E", [("N", 1), ("E", neg_emb_inputs.size("E"))])
        elif neg_emb_inputs.dim() > 3:
            raise ValueError("Dimension of neg_emb_inputs can only be 2 or 3.")

        # calculate prediction of prediction subnet
        # concatenate user and content on dimension of N, which return shape = (B, N, E)
        # then flatten to shape = (B, N * E)
        # return y_pred with shape = (B, O = 1)
        cat_pred = torch.cat([user_emb_inputs, content_emb_inputs], dim="N")
        cat_pred = cat_pred.flatten(["N", "E"], "E")
        y_pred = self.prediction(cat_pred)

        # calculate inference of matching subnet,
        # return (user_match, item_match) with shape = (B, O)
        # then multiply them and sum on dimension O, which return shape = (B, O)
        # finally apply sigmoid to y_pred
        user_match = self.matching["user"](user_emb_inputs.flatten(["N", "E"], "E"))
        item_match = self.matching["item"](content_emb_inputs.flatten(["N", "E"], "E"))
        y_match = (user_match * item_match).sum(dim="O", keepdim=True)
        y_match = self.matching["sigmoid"](y_match)

        # calculate features' representations of items and return shape = (B, 1 or N, O_C)
        content_corr = self.correlation(content_emb_inputs)
        positive_corr = self.correlation(pos_emb_inputs)
        negative_corr = self.correlation(neg_emb_inputs)
        
        # calculate the inference of correlation subnet between content 
        # and positive or negative by dot-product, and return a pair of 
        # tensors with shape = (B, 1)
        y_corr_pos = (content_corr * positive_corr).sum(dim="O")
        y_corr_neg = (content_corr * negative_corr).sum(dim="O").mean(dim="N", keepdim=True)

        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        y_pred = y_pred.rename(None)
        y_match = y_match.rename(None)
        y_corr_pos = y_corr_pos.rename(None)
        y_corr_neg = y_corr_neg.rename(None)

        return y_pred, y_match, y_corr_pos, y_corr_neg
