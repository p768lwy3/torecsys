from . import _Inputs
from torecsys.functional import show_attention, dummy_attention
from torecsys.utils.decorator import jit_experimental, no_jit_experimental
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class ListIndicesEmbedding(_Inputs):
    r"""Base Inputs class for embedding of list of indices without order, which embed the 
    list by multihead attention and aggregate before return.
    """
    @jit_experimental
    def __init__(self,
                 embed_size    : int,
                 field_size    : int,
                 padding_idx   : int  = 0,
                 use_attn      : bool = True,
                 output_method : str  = "avg_pooling",
                 nn_embedding  : nn.Parameter = None,
                 **kwargs):
        r"""Initialize ListIndicesEmbedding.
        
        Args:
            embed_size (int): Size of embedding tensor
            field_size (int): Size of inputs field
            padding_idx (int, optional): Padding index. 
                Defaults to 0.
            use_attn (bool, optional): Whether multihead attention is used or not.  
                Defaults to True.
            output_method (str, optional): Method of aggregation. 
                Allow: ["avg_pooling", "max_pooling", "mean", "sum"]. 
                Defaults to "avg_pooling".
            nn_embedding (nn.Parameter, optional): Pretrained embedding values. 
                Defaults to None.
        
        Kwargs:
            num_heads (int): Number of heads for MultiheadAttention.
                Required when use_attn is True.
            dropout (float, optional): Probability of Dropout in MultiheadAttention. 
                Default = 0.0.
            bias (bool, optional): Whether bias is added to multihead attention or not. 
                Default = True.
            add_bias_kv (bool, optional): Whether bias is added to the key and value 
                sequences at dim = 1 in multihead attention or not. 
                Default = False.
            add_zero_attn (bool, optional): Whether a new batch of zeros is added to 
                the key and value sequences at dim = 1 in multihead attention  or not. 
                Default = False.
        
        Attributes:
            length (int): Size of embedding tensor.
            embed_size (int): Size of embedding tensor.
            field_size (int): Size of inputs' field.
            embedding (torch.nn.Module): Embedding layer.
            use_attn (bool): Flag to show attention is used or not.
            attn_args (dict): Dictonary of arguments used in MultiheadAttention.
            attention (Union[torch.nn.Module, callable]): MultiheadAttention layer or dummy_attention.
            aggregation (Union[torch.nn.Module, callable]): Pooling layer or aggregation function.
            output_method (string): Type of output_method.
            
        Raises:
            ValueError: when output_method is not in ["avg_pooling", "max_pooling", "mean", "sum"].
        """
        # refer to parent class
        super(ListIndicesEmbedding, self).__init__()

        # bind embedding to pre-trained embedding module if nn_embedding is not None
        if nn_embedding is not None:
            self.embed_size = nn_embedding.size(1)
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        # else, create a embedding module with the given arguments
        else:
            self.embed_size = embed_size
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx)
        
        # bind field_size and length to field_size and embed_size
        self.field_size = field_size
        self.length = self.embed_size

        # initialize multihead attention layer and bind use_attn to use_attn
        self.use_attn = use_attn
        if self.use_attn:
            # parse arguments of multihead attention layer
            self.attn_args = dict(
                embed_dim     = embed_size,
                num_heads     = kwargs.get("num_heads"),
                dropout       = kwargs.get("dropout", 0.0),
                bias          = kwargs.get("bias", True),
                add_bias_kv   = kwargs.get("add_bias_kv", False),
                add_zero_attn = kwargs.get("add_zero_attn", False)
            )
            self.attention = nn.MultiheadAttention(**self.attn_args)
        else:
            # bind attention to a dummy function called dummy_attention 
            # which will return input key directly
            self.attention = dummy_attention
        
        # initialize aggregation layer for outputs and bind output_method to output_method
        __output_method__ = ["avg_pooling", "max_pooling", "mean", "none", "sum"]
        if output_method == "avg_pooling":
            self.aggregation = nn.AdaptiveAvgPool1d(1)
        elif output_method == "max_pooling":
            self.aggregation = nn.AdaptiveMaxPool1d(1)
        elif output_method == "mean":
            self.aggregation = partial(torch.mean, dim=1, keepdim=True)
        elif output_method == "none":
            self.aggregation = torch.Tensor
        elif output_method == "sum":
            self.aggregation = partial(torch.sum, dim=1, keepdim=True)
        else:
            raise ValueError("output_method only allows [%s]." % (", ".join(__output_method__)))
        self.output_method = output_method

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward calculation of ListIndicesEmbedding.

        Args:
            inputs (T), shape = (B, L), dtype = torch.long: List of tensor of indices in inputs fields.
        
        Returns:
            Tuple[T, T], shape = ((B, 1 or L, E), (B, L, L) or (None)), 
                dtype = (torch.float, torch.float): Outputs of ListIndicesEmbedding and Attention weights.
        """
        # get embedding and output shape = (B, L, E) 
        # then, transpose them to (L, B, E)
        outputs = self.embedding(inputs)
        outputs = outputs.transpose(0, 1)

        # compute self-attention, and outputs' shape = (B, L, E) 
        # and attn_weights' shape = (B, L, L) if use_attn = True
        # then transpose back to (B, L, E)
        outputs, attn_weights = self.attention(outputs, outputs, outputs)
        outputs = outputs.transpose(0, 1)
        
        # calculate aggregation of outputs
        if self.output_method == "avg_pooling" or self.output_method == "max_pooling":
            # transpose from (B, L, E) to (B, E, L)
            outputs = outputs.transpose(1, 2)
            # shape of outputs = (B, E, 1)
            outputs = self.aggregation(outputs)
            # transpose from (B, E, 1) to (B, 1, E)
            outputs = outputs.transpose(1, 2)
        else:
            # outputs' shape = (B, 1, E) if output_method == "mean" or "sum"
            # else outputs' shape = (B, L, E) if output_method == "none"
            outputs = self.aggregation(outputs)
        return outputs, attn_weights

    @no_jit_experimental
    def show_attention(self, 
                       inputs  : torch.Tensor, 
                       savedir : str = None):
        r"""Show heatmap of self-attention in multihead attention.
        
        Args:
            inputs (T), shape = (1, L), dtype = torch.long: A single sample of list of tensor of 
                indices in inputs fields.
            savedir (str, optional): Directory to save heatmap. Defaults to None.
        
        Raises:
            ValueError: when batch size is not equal to 1.
            ValueError: when self.attn is not True
        """
        # only can be called when use_attn is True
        if self.use_attn:
            # check whether batch size is equal to 1 or not.
            if inputs.size(0) != 1:
                raise ValueError("batch size must be equal to 1.")
            
            # set torch to no_grad for inference
            with torch.no_grad():
                # get embedding vectors and transpose from (1, L, E) to (L, 1, E)
                outputs = self.embedding(inputs)
                outputs = outputs.transpose(0, 1)
                
                # calcualte self-attentions and attn_weights' shape = (1, L, L)
                _, attn_weights = self.attention(outputs, outputs, outputs)
            
            # flatten the attentions by removing 1st dimension
            attention = np.squeeze(attn_weights.numpy(), axis=0)

            # create a list of string of index from inputs to be the axis of plot
            axis = [str(x) for x in inputs.squeeze().tolist()]

            # show attentions with a heatmap plot
            show_attention(attention, xaxis=axis, yaxis=axis, savedir=savedir)
        else:
            raise ValueError("show_attention cannot be called if use_attn is False.")
