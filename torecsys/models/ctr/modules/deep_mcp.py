from . import _CtrModule
from ..layers import MultilayerPerceptronLayer
from torecsys.utils.logging.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Dict, List


# in-development
class DeepMatchingCorrelationPredictionModule(_CtrModule):
    r"""DeepMacthingCorrelationPredictionModule is a module of Deep Matching, Correlation and 
    Preidction (DeepMCP) proposed by Wentao Ouyang et al of Alibaba Group in 2019, which is a 
    model concatenated three parts: Matching, Correlation, and Predction, to adjust the distance 
    between user-item and item-item in the following way: 

    #. Prediction subnet: Feed-forward Dense Layers

    #. Matching subnt: sigmoid of dot-product between high-level representations of users and items,
    with the following calculation: :math"`\^{y} = \frac{1}{1 + \text{exp}(-(w^{T}z + b))}` .

    #. Correlation subnet: 

    :Reference:

    #. `Wentao Ouyang et al, 2019. Representation Learning-Assisted Click-Through Rate Prediction <https://arxiv.org/pdf/1906.04365.pdf>`

    """

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
        r"""initialize DeepMCP Module
        
        Args:
            embed_size (int): embedding size
            user_num_fields (int): number of fields in users' inputs
            item_num_fields (int): number of fields in items' inputs
            corr_output_size (int): output size of multilayer perceptron in correlation subnet
            match_output_size (int): output size of multilayer perceptron in matching subnet
            corr_layer_sizes (List[int]): layer sizes of multilayer perceptron in correlation subnet
            match_layer_sizes (List[int]): layer sizes of multilayer perceptron in matching subnet
            pred_layer_sizes (List[int]): layer sizes of multilayer perceptron in prediction subnet
            corr_dropout_p (List[float], optional): dropout probability after activation of each layer in multilayer perceptron in correlation subnet. Defaults to None.
            match_dropout_p (List[float], optional): dropout probability after activation of each layer in multilayer perceptron in matching subnet. Defaults to None.
            pred_dropout_p (List[float], optional): dropout probability after activation of each layer in multilayer perceptron in prediction subnet. Defaults to None.
            corr_activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function of each layer in multilayer perceptron of correlation subnet. Allow: [None, Callable[[torch.Tensor], torch.Tensor]]. Defaults to nn.ReLU().
            match_activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function of each layer in multilayer perceptron of matching subnet. Allow: [None, Callable[[torch.Tensor], torch.Tensor]]. Defaults to nn.ReLU().
            pred_activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function of each layer in multilayer perceptron of prediction subnet. Allow: [None, Callable[[torch.Tensor], torch.Tensor]]. Defaults to nn.ReLU().
        """
        super(DeepMatchingCorrelationPredictionModule, self).__init__()

        # initialize prediction subnet, which is a dense network with output's size = 1
        self.prediction = MultilayerPerceptronLayer(
            output_size = 1,
            layer_sizes = pred_layer_sizes,
            embed_size  = embed_size,
            num_fields  = (user_num_fields + item_num_fields),
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
            MultilayerPerceptronLayer(
                output_size = match_output_size,
                layer_sizes = match_layer_sizes,
                embed_size  = embed_size,
                num_fields  = user_num_fields,
                dropout_p   = match_dropout_p,
                activation  = match_activation
            )
        )
        self.matching["user"].add_module(nn.Tanh())
        
        # initialize item part of matching subnet 
        self.matching["item"] = nn.Sequential()
        self.matching["item"].add_module(
            MultilayerPerceptronLayer(
                output_size = match_output_size,
                layer_sizes = match_layer_sizes,
                embed_size  = embed_size,
                num_fields  = item_num_fields,
                dropout_p   = match_dropout_p,
                activation  = match_activation
            )
        )
        self.matching["item"].add_module(nn.Tanh())

        # initialize correlation subnet
        self.correlation = nn.ModuleDict()
        self.correlation["item"] = nn.Sequential()
        self.correlation["item"].add_module(
            MultilayerPerceptronLayer(
                output_size = corr_output_size,
                layers_size = corr_layer_sizes,
                embed_size  = embed_size,
                num_fields  = item_num_fields,
                dropout_p   = corr_dropout_p,
                activation  = corr_activation
            )
        )
        self.correlation["item"].add_module(nn.Tanh())

    def forward(self, inputs: Dict[torch.Tensor]) -> Tuple[torch.Tensor]:
        r"""feed forward calculation of DeepMCP
        
        Args:
            inputs (Dict[torch.Tensor]): [description]
        
        Returns:
            Tuple[torch.Tensor]: [description]
        """

        # calculate prediction of prediction subnet, return shape = (batch size, 1)
        cat_pred = torch.cat([inputs["user"], inputs["content_item"]], dim=1)
        y_pred = self.prediction(cat_pred)

        # calculate inference of matching subnet, return shape = (batch size, 1)
        user_match = self.matching["user"](inputs["user"])
        item_match = self.matching["item"](inputs["item"])
        y_match = (user_match * item_match).sum(dim=1, keepdim=True)
        y_match = self.matching["sigmoid"](y_match)

        # calculate inference of correlation subnet between content and 
        # positive or negative, return a pair of tensors with shape = (batch size, 1)
        content_corr = self.correlation["item"](inputs["item"])
        positive_corr = self.correlation["item"](inputs["positive"])
        negative_corr = self.correlation["item"](inputs["negative"])
        
        # calculate the inference by dot-product
        ## need to modify output shape of mlp to (B, 1, E)
        y_corr_pos = (content_corr * positive_corr).sum(dim=2)
        ## need to modify mlp layers for the negative inputs with shape = (B, nneg, field * emb)
        y_corr_neg = (content_corr * negative_corr).sum(dim=2).mean(dim=1, keepdim=True)

        return y_pred, y_match, y_corr_pos, y_corr_neg
