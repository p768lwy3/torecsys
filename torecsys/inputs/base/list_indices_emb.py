from . import _Inputs
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torecsys.utils.decorator import no_jit_experimental, no_jit_experimental_by_namedtensor
from torecsys.utils.operations import dummy_attention, show_attention
from typing import Tuple

class ListIndicesEmbedding(_Inputs):
    r"""Base Inputs class for embedding of list of indices without order, which embed the 
    list by multihead attention and aggregate before return.
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size    : int,
                 field_size    : int,
                 padding_idx   : int  = 0,
                 use_attn      : bool = False,
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
                Defaults to False.
            output_method (str, optional): Method of aggregation. 
                Allow: ["avg_pooling", "max_pooling", "mean", "none", "sum"]. 
                Defaults to "avg_pooling".
            nn_embedding (nn.Parameter, optional): Pretrained embedding values. 
                Defaults to None.
        
        Kwargs:
            num_heads (int): Number of heads for MultiheadAttention.
                Required when use_attn is True. Default to 1.
            dropout (float, optional): Probability of Dropout in MultiheadAttention. 
                Default to 0.0.
            bias (bool, optional): Whether bias is added to multihead attention or not. 
                Default to True.
            add_bias_kv (bool, optional): Whether bias is added to the key and value 
                sequences at dim = 1 in multihead attention or not. 
                Default to False.
            add_zero_attn (bool, optional): Whether a new batch of zeros is added to 
                the key and value sequences at dim = 1 in multihead attention  or not. 
                Default to False.
        
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

        # bind use_attn to use_attn
        self.use_attn = use_attn

        if self.use_attn:
            # parse arguments of multihead attention layer
            self.attn_args = dict(
                embed_dim     = embed_size,
                num_heads     = kwargs.get("num_heads", 1),
                dropout       = kwargs.get("dropout", 0.0),
                bias          = kwargs.get("bias", True),
                add_bias_kv   = kwargs.get("add_bias_kv", False),
                add_zero_attn = kwargs.get("add_zero_attn", False)
            )
            # initialize multihead attention layer
            self.attention = nn.MultiheadAttention(**self.attn_args)
        else:
            # bind attention to a dummy function called dummy_attention 
            # which will return input key directly
            self.attention = dummy_attention
        
        # initialize aggregation layer for outputs and bind output_method to output_method
        if output_method == "avg_pooling":
            self.aggregation = nn.AdaptiveAvgPool1d(1)
        elif output_method == "max_pooling":
            self.aggregation = nn.AdaptiveMaxPool1d(1)
        elif output_method == "mean":
            self.aggregation = partial(torch.mean, dim="N", keepdim=True)
        elif output_method == "none":
            self.aggregation = torch.Tensor
        elif output_method == "sum":
            self.aggregation = partial(torch.sum, dim="N", keepdim=True)
        else:
            raise ValueError('output_method only allows ["avg_pooling", "max_pooling", "mean", "none", "sum"].')
        self.output_method = output_method

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward calculation of ListIndicesEmbedding.

        Args:
            inputs (T), shape = (B, L), dtype = torch.long: List of tensor of indices in inputs fields.
        
        Returns:
            Tuple[T, T], shape = ((B, 1 or L, E), (B, L, L) or (None)), 
                dtype = (torch.float, torch.float): Outputs of ListIndicesEmbedding and Attention weights.
        """
        # Get and reshape embedding tensors
        # inputs: inputs, shape = (B, L)
        # output: outputs, shape = (L, B, E)
        outputs = self.embedding(inputs.rename(None))
        outputs.names = ("B", "L", "E")
        outputs = outputs.align_to("L", "B", "E")

        # Compute self-attention and reshape output of attention
        # inputs: outputs, shape = (L, B, E)
        # output: outputs, shape = (B, L, E)
        outputs = outputs.rename(None)
        outputs, _ = self.attention(outputs, outputs, outputs)
        outputs.names = ("L", "B", "E")
        outputs = outputs.align_to("B", "L", "E")
        
        # Calculate aggregation of outputs
        if self.output_method == "avg_pooling" or self.output_method == "max_pooling":
            # transpose outputs
            # inputs: outputs, shape = (B, L, E)
            # output: outputs, shape = (B, E, L)
            outputs = outputs.align_to("B", "E", "L")

            # apply pooling to outputs if batch size > 1
            # inputs: outputs, shape = (B, E, L)
            # output: outputs, shape = (B, E, N = 1)
            outputs = self.aggregation(outputs.rename(None)) if outputs.size("B") > 1 else outputs
            outputs.names = ("B", "E", "N")

            # transpose outputs
            # inputs: outputs, shape = (B, E, N)
            # output: outputs, shape = (B, N, E)
            outputs = outputs.align_to("B", "N", "E")

        else:
            # apply aggregation function to outputs
            # inputs: outputs, shape = (B, L, E)
            # output: outputs, shape = (B, 1, E) if output_method in ["mean", "sum"] else (B, L, E)
            outputs = self.aggregation(outputs.rename(None))
            outputs.names = ("B", "N", "E")

        return outputs

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
            if inputs.size("B") != 1:
                raise ValueError("batch size must be equal to 1.")
            
            # set torch to no_grad for inference
            with torch.no_grad():
                # Get and reshape embedding tensors
                # inputs: inputs, shape = (B, L)
                # output: outputs, shape = (L, B, E)
                outputs = self.embedding(inputs.rename(None))
                outputs.names = ("B", "L", "E")
                outputs = outputs.align_to("L", "B", "E")
                
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

            # Show attentions with a heatmap plot
            show_attention(attn_weights, xaxis=axis, yaxis=axis, savedir=savedir)
        else:
            raise ValueError("show_attention cannot be called if use_attn is False.")
        