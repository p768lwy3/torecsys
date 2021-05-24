from typing import Optional

import torch
import torch.nn as nn

from torecsys.layers import PositionEmbeddingLayer
from torecsys.models.ltr import ReRankingModel


class PersonalizedReRankingModel(ReRankingModel):
    """
    Model class of Personalized Re-ranking Model (PRM).

    Personalized Re-ranking model is proposed by Alibaba Group in 2019, which is to re-ranking
    results from a ranked model with extra information, including user preference from another
    pre-trained model and features from crossing items.

    The main mechanism of PRM are: 

    #. to concatenate user-features before encoding layers (i.e. transformer part).

    #. to use Transformer to capture the features without considering sequence order.

    Finally, sort items list by scores calculated by softmax layer at the end.

    :Reference:

    #. `Changhua Pei et al, 2019. Personalized Re-ranking for Recommendation <https://arxiv.org/abs/1904.06813>`_.

    """

    def __init__(self,
                 embed_size: int,
                 max_num_position: int,
                 encoding_size: int,
                 num_heads: int,
                 num_layers: int,
                 use_bias: Optional[bool] = True,
                 dropout: Optional[float] = None,
                 **kwargs):
        """
        Initialize PersonalizedReRankingModel
        
        Args:
            embed_size (int): size of embedding tensor
            max_num_position (int): maximum length of list, i.e. Maximum number of position
            encoding_size (int): size of input of encoding layer
            num_heads (int): number of heads in Multi Head Attention
            num_layers (int): number of layers of Transformer
            use_bias (bool, optional): boolean flag to use bias. Default to True
            dropout (float, optional): probability of Dropout in Multi Head Attention. Defaults to None

        Arguments:
            fnn_dropout_p (float): probability of Dropout in feed-forward of Transformer.
            fnn_activation (nn.Module): activation function of feed-forward of Transformer.
        """
        super().__init__()

        self.layers = nn.ModuleDict()

        self.layers['InputLayer'] = nn.ModuleDict()
        self.layers['InputLayer']['PositionEmbedding'] = PositionEmbeddingLayer(max_num_position=max_num_position) \
            if use_bias else None
        self.layers['InputLayer']['FeedForward'] = nn.Linear(embed_size, encoding_size)

        self.layers['EncodingLayer'] = nn.ModuleDict()
        for i in range(num_layers):
            layer = nn.ModuleDict()

            multi_head = nn.MultiheadAttention(encoding_size, num_heads, dropout)
            layer['MultiHeadAttention'] = multi_head

            batchnorm = nn.BatchNorm1d(max_num_position)
            layer['AttentionBatchNorm'] = batchnorm

            feedforward = nn.Sequential()
            feedforward.add_module('FeedForward', nn.Linear(encoding_size, encoding_size))

            if kwargs.get('fnn_activation'):
                feedforward.add_module('Activation', kwargs.get('fnn_activation'))
            else:
                feedforward.add_module('Activation', nn.ReLU())

            if kwargs.get('fnn_dropout_p'):
                feedforward.add_module('Dropout', nn.Dropout(kwargs.get('fnn_dropout_p')))

            layer['FeedForward'] = feedforward
            layer['FNNBatchNorm'] = nn.BatchNorm1d(max_num_position)

            self.layers['EncodingLayer'][f'Transformer_{i}'] = layer

        self.layers['OutputLayer'] = nn.ModuleDict()
        self.layers['OutputLayer']['FeedForward'] = nn.Linear(encoding_size, 1)
        self.layers['OutputLayer']['Softmax'] = nn.Softmax(dim=1)

    def forward(self, feat_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of PersonalizedReRankingModel
        
        Args:
            feat_inputs (T), shape = (B, L, E), data_type = torch.int: features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of PersonalizedReRankingModel.
        """
        # 1) Input Layer Part
        # Add up feat_inputs and positional embedding
        # inputs: feat_inputs, shape = (B, L, E)
        # inputs: layers["InputLayer"]["PositionEmbedding"], shape = (B, L, E)
        # output: output, shape = (B, L, E)
        if self.layers['InputLayer']['PositionEmbedding'] is not None:
            output = self.layers['InputLayer']['PositionEmbedding'](feat_inputs)
        else:
            output = feat_inputs

        # Calculate forwardly with feed-forward layer of input layer
        # inputs: output, shape = (B, L, E)
        # output: output, shape = (B, L, E')
        output = self.layers['InputLayer']['FeedForward'](output)

        # 2) Encoding Layer Part
        for i in range(len(self.layers['EncodingLayer'])):
            # Get layer from model
            layer = self.layers['EncodingLayer'][f'Transformer_{i}']

            # Copy output as residual
            residual_a = output

            # Compute Multi Head attention
            # inputs: output, shape = (B, L, E)
            # output: output, shape = (B, L, E)
            output.names = ('B', 'L', 'E',)
            output = output.align_to('L', 'B', 'E')
            output.names = None

            output, _ = layer["MultiHeadAttention"](output, output, output)

            output.names = ('L', 'B', 'E',)
            output = output.align_to('B', 'L', 'E')
            output.names = None

            # Add residual and apply batchnorm
            # inputs: output, shape = (B, L, E)
            # inputs: input_i, shape = (B, L, E)
            # output: output, shape = (B, L, E)
            output = output + residual_a
            output = layer['AttentionBatchNorm'](output.rename(None))

            # Copy output as residual
            residual_f = output

            # Calculate forwardly with feed-forward
            # inputs: output, shape = (B, L, E)
            # output: output, shape = (B, L, E)
            output = layer['FeedForward'](output)

            # Add residual and apply batchnorm
            # inputs: output, shape = (B, L, E)
            # inputs: input_i, shape = (B, L, E)
            # output: output, shape = (B, L, E)
            output = output + residual_f
            output = layer['FNNBatchNorm'](output.rename(None))

        # 3) Output Layer Part
        # Calculate forwardly with feed-forward
        # inputs: output, shape = (B, L, E)
        # output: output, shape = (B, L, O = 1)
        output = self.layers['OutputLayer']['FeedForward'](output)
        output.names = ('B', 'L', 'O',)

        # Flatten output
        # inputs: output, shape = (B, L, O)
        # output: output, shape = (B, O = L)
        output = output.flatten(('L', 'O',), 'O')

        # Apply softmax
        # output: output, shape = (B, O = L)
        # output: output, shape = (B, O = L)
        output = self.layers['OutputLayer']['Softmax'](output)

        return output
