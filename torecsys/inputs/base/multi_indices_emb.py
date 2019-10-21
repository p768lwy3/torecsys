from . import _Inputs
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import numpy as np
import torch
import torch.nn as nn
from typing import List


class MultiIndicesEmbedding(_Inputs):
    r"""Base Inputs class for embedding indices in multi fields of inputs, which is more 
    efficent than embedding with a number of SingleIndexEmbedding.
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 embed_size   : int,
                 field_sizes  : List[int],
                 flatten      : bool = False,
                 nn_embedding : nn.Parameter = None,
                 device       : str = "cpu",
                 **kwargs):
        r"""Initialize MultiIndicesEmbedding.
        
        Args:
            embed_size (int): Size of embedding tensor
            field_sizes (List[int]): List of inputs fields' sizes
            flatten (bool, optional): Whether outputs is reshape to (B, 1, N * E) or not before return.
                Defaults to False.
            nn_embedding (nn.Parameter, optional): Pretrained embedding values. 
                Defaults to None.
            device (str): Device of torch.
                Defaults to cpu.
        
        Arguments:
            length (int): Size of embedding tensor multiply by number of fields if flatten is 
                True, else Size of embedding tensor.
            embedding (torch.nn.Module): Embedding layer.
            flatten (bool): Flag to show outputs will be flatten or not.
            offsets (T): Tensor of offsets to adjust values of inputs to fit the indices of 
                embedding tensors.
        """
        # refer to parent class
        super(MultiIndicesEmbedding, self).__init__()

        # bind embedding to pre-trained embedding module if nn_embedding is not None
        if nn_embedding is not None:
            embed_size = nn_embedding.size(1)
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        # else, create a embedding module with the given arguments
        else:
            self.embedding = nn.Embedding(sum(field_sizes), embed_size, **kwargs)

        # create offsets to re-index inputs by adding them up
        self.offsets = torch.Tensor((0, *np.cumsum(field_sizes)[:-1])).long().unsqueeze(0)
        self.offsets.names = ("B", "N")
        self.offsets.to(device)

        # bind length to embed_size * length of field_sizes (i.e. num_fields) if flatten is True
        if flatten:
            self.length = embed_size * len(field_sizes)
        # else bind length to embed_size
        else:
            self.length = embed_size
        
        # bind flatten to flatten
        self.flatten = flatten

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of MultiIndicesEmbedding
        
        Args:
            inputs (T), shape = (B, N), dtype = torch.long: Tensor of indices in inputs fields.
        
        Returns:
            T, shape = (B, 1, N * E) or (B, N, E), dtype = torch.float: Outputs of MultiIndicesEmbedding.
        """
        # add offset to adjust values of inputs to fit the indices of embedding tensors
        inputs = inputs + self.offsets
        outputs = self.embedding(inputs.rename(None))

        # since the name will be removed after embedding
        # set the name again to use .size() below.
        outputs.names = ("B", "N", "E") 

        # flatten to (B, 1, N * E) if flatten is True
        if self.flatten:
            batch_size = outputs.size("B")
            outputs = outputs.flatten(["N", "E"], "E").rename(None).unsqueeze(1)
            ## outputs = outputs.view(batch_size, 1, -1)
        
        # else outputs' shape = (B, N, E)
        # set names to the tensor
        outputs.names = ("B", "N", "E")
        return outputs
        