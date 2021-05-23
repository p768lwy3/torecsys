from functools import partial
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput
from torecsys.utils.operations import dummy_attention, show_attention


class ListIndicesEmbedding(BaseInput):
    """
    Base Input class for embedding of list of indices without order, which embed the list
    by multi head attention and aggregate before return
    """

    def __init__(self,
                 embed_size: Optional[int] = None,
                 field_size: Optional[int] = None,
                 padding_idx: Optional[int] = 0,
                 nn_embedding: Optional[nn.Parameter] = None,
                 use_attn: Optional[bool] = False,
                 output_method: Optional[str] = 'avg_pooling',
                 **kwargs):
        """
        Initialize ListIndicesEmbedding.
        
        Args:
            embed_size (int, optional): size of embedding tensor. Defaults to None
            field_size (int, optional): size of inputs field. Defaults to None
            padding_idx (int, optional): padding index. Defaults to 0
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None
            use_attn (bool, optional): whether multi head attention is used or not. Defaults to False
            output_method (str, optional): method of aggregation.
                allow: ["avg_pooling", "max_pooling", "mean", "none", "sum"].
                Defaults to "avg_pooling"

        Kwargs:
            num_heads (int): number of heads for multi head attention. Required when use_attn is True.
                Default to 1
            dropout (float, optional): probability of Dropout in multi head attention. Default to 0.0
            bias (bool, optional): Whether bias is added to multi head attention or not. Default to True
            add_bias_kv (bool, optional): Whether bias is added to the key and value sequences at dim = 1
                in multi head attention or not. Default to False
            add_zero_attn (bool, optional): Whether a new batch of zeros is added to the key and value sequences
                at dim = 1 in multi head attention or not. Default to False
        
        Attributes:
            length (int): size of embedding tensor
            embed_size (int): size of embedding tensor
            field_size (int): size of inputs' field
            padding_idx (int): padding index of the embedder
            embedding (torch.nn.Module): embedding layer
            use_attn (bool): flag to show attention is used or not
            attn_args (dict): dictionary of arguments used in multi head attention
            attention (Union[torch.nn.Module, callable]): multi head attention layer or dummy_attention
            aggregation (Union[torch.nn.Module, callable]): pooling layer or aggregation function
            output_method (string): type of output_method
            
        Raises:
            ValueError: when output_method is not in ["avg_pooling", "max_pooling", "mean", "sum"]
        """
        super().__init__()

        if nn_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        elif field_size is not None and embed_size is not None:
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx)
        else:
            raise ValueError('missing required arguments')

        self.field_size = self.embedding.num_embeddings
        self.embed_size = self.embedding.embedding_dim
        self.padding_idx = self.embedding.padding_idx
        self.length = self.embed_size

        self.use_attn = use_attn
        if self.use_attn:
            self.attn_args = {
                'embed_dim': self.embed_size,
                'num_heads': kwargs.get('num_heads', 1),
                'dropout': kwargs.get('dropout', 0.0),
                'bias': kwargs.get('bias', True),
                'add_bias_kv': kwargs.get('add_bias_kv', False),
                'add_zero_attn': kwargs.get('add_zero_attn', False)
            }
            self.attention = nn.MultiheadAttention(**self.attn_args)
        else:
            self.attention = dummy_attention

        # initialize aggregation layer for outputs and bind output_method to output_method
        if output_method == 'avg_pooling':
            self.aggregation = nn.AdaptiveAvgPool1d(1)
        elif output_method == 'max_pooling':
            self.aggregation = nn.AdaptiveMaxPool1d(1)
        elif output_method == 'mean':
            self.aggregation = partial(torch.mean, dim='N', keepdim=True)
        elif output_method == 'none':
            self.aggregation = torch.Tensor
        elif output_method == 'sum':
            self.aggregation = partial(torch.sum, dim='N', keepdim=True)
        else:
            raise ValueError('output_method only allows ["avg_pooling", "max_pooling", "mean", "none", "sum"].')
        self.output_method = output_method

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward calculation of ListIndicesEmbedding.

        Args:
            inputs (T), shape = (B, L), data_type = torch.long: list of tensor of indices in inputs fields.
        
        Returns:
            Tuple[T, T], shape = ((B, 1 or L, E), (B, L, L) or (None)), 
                data_type = (torch.float, torch.float): outputs of ListIndicesEmbedding and Attention weights.

        TODO: it will raise error now if inputs contains any empty lists. Planning to add padding idx to prevent the
            error.
        """
        # Get and reshape embedding tensors
        # inputs: inputs, shape = (B, L)
        # output: outputs, shape = (L, B, E)
        outputs = self.embedding(inputs.rename(None))
        outputs.names = ('B', 'L', 'E',)
        outputs = outputs.align_to('L', 'B', 'E')

        # Compute self-attention and reshape output of attention
        # inputs: outputs, shape = (L, B, E)
        # output: outputs, shape = (B, L, E)
        outputs = outputs.rename(None)
        outputs, _ = self.attention(outputs, outputs, outputs)
        outputs.names = ('L', 'B', 'E',)
        outputs = outputs.align_to('B', 'L', 'E')

        # Calculate aggregation of outputs
        if self.output_method == 'avg_pooling' or self.output_method == 'max_pooling':
            # transpose outputs
            # inputs: outputs, shape = (B, L, E)
            # output: outputs, shape = (B, E, L)
            outputs = outputs.align_to('B', 'E', 'L')

            # apply pooling on outputs
            # inputs: outputs, shape = (B, E, L)
            # output: outputs, shape = (B, E, N = 1)
            outputs = self.aggregation(outputs.rename(None))
            outputs.names = ('B', 'E', 'N',)

            # transpose outputs
            # inputs: outputs, shape = (B, E, N)
            # output: outputs, shape = (B, N, E)
            outputs = outputs.align_to('B', 'N', 'E')

        else:
            # apply aggregation function to outputs
            # inputs: outputs, shape = (B, L, E)
            # output: outputs, shape = (B, 1, E) if output_method in ["mean", "sum"] else (B, L, E)
            outputs = self.aggregation(outputs.rename(None))
            outputs.names = ('B', 'N', 'E',)

        return outputs

    def show_attention(self,
                       inputs: torch.Tensor,
                       save_dir: Optional[str] = None):
        """
        Show heat map of self-attention in multi head attention.
        
        Args:
            inputs (T), shape = (1, L), data_type = torch.long: a single sample of list of tensor of indices
                in inputs fields
            save_dir (str, optional): directory to save heat map. Defaults to None
        
        Raises:
            ValueError: when batch size is not equal to 1
            ValueError: when self.attn is not True
        """
        # only can be called when use_attn is True
        if self.use_attn:
            # check whether batch size is equal to 1 or not.
            if inputs.size('B') != 1:
                raise ValueError('batch size must be equal to 1')

            # set torch to no_grad for inference
            with torch.no_grad():
                # Get and reshape embedding tensors
                # inputs: inputs, shape = (B, L)
                # output: outputs, shape = (L, B, E)
                outputs = self.embedding(inputs.rename(None))
                outputs.names = ('B', 'L', 'E',)
                outputs = outputs.align_to('L', 'B', 'E')

                # Compute self-attention and reshape output of attention
                # inputs: outputs, shape = (L, B, E)
                # output: attn_weights, shape = (1, L, L)
                outputs = outputs.rename(None)
                _, attn_weights = self.attention(outputs, outputs, outputs)

            # Remove dim 1 to flatten attn_weights and convert to np.array
            # inputs: attn_weights, shape = (1, L, L)
            # output: attn_weights, shape = (L, L)
            attn_weights = np.squeeze(attn_weights.numpy(), axis=0)

            # Create a list of string of index from inputs to be the axis of plot
            axis = [str(x) for x in inputs.rename(None).squeeze().tolist()]

            # Show attentions with a heat map plot
            show_attention(attn_weights, x_axis=axis, y_axis=axis, save_dir=save_dir)
        else:
            raise ValueError('show_attention cannot be called if use_attn is False.')
