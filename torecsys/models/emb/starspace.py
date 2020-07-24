from functools import partial
from typing import Callable, Tuple

import torch

from torecsys.layers import StarSpaceLayer
from torecsys.utils.operations import inner_product_similarity
from . import _EmbModel


class StarSpaceModel(_EmbModel):
    r"""Model class of StatSpaceModel.

    Starspace is a model proposed by Facebook in 2017, which to embed relationship between 
    different kinds of pairs, like (word, tag), (user-group) etc, by calculating similarity 
    between context and positive samples or negative samples, and margin ranking loss between 
    similarities.

    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`_.
    
    """

    def __init__(self,
                 embed_size: int,
                 num_neg: int,
                 similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = partial(inner_product_similarity,
                                                                                            dim=2)):
        r"""Initialize StarSpaceModel
        
        Args:
            similarity (Callable[[T, T], T], optional): Function of similarity between two tensors. 
                e.g. torch.nn.functional.cosine_similarity. 
                Defaults to partial(inner_product_similarity, dim=2).
        
        Attributes:
            embed_size (int): Size of embedding tensor
            num_neg (int): Number of negative samples
            starspace (nn.Module): Module of starspace layer.
        """
        # Refer to parent class
        super(StarSpaceModel, self).__init__()

        # Bind embed_size and num_neg to embed_size and num_neg respectively
        self.embed_size = embed_size
        self.num_neg = num_neg

        # Initialize starspace layer
        self.starspace = StarSpaceLayer(similarity)

    def forward(self,
                context_inputs: torch.Tensor,
                target_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward calculation of StarSpaceModel
        
        Args:
            context_inputs (T), shape = (B * (1 + Nneg), 1, E), dtype = torch.float: Embedded features tensors of context
            target_inputs (T), shape = (B * (1 + Nneg), 1, E), dtype = torch.float: Embedded features tensors of target
        
        Returns:
            Tuple[T, T], shape = (B, 1 / Nneg): Output of StarSpaceModel.
        """
        # Get batch size from context_inputs
        agg_batch_size = context_inputs.size("B")
        batch_size = int(agg_batch_size // (1 + self.num_neg))

        # Reshape context_inputs 
        # inputs: context_inputs, shape = (B * (1 + Nneg), 1, E)
        # output: context_inputs, shape = (B, (1 + Nneg), E)
        context_inputs = context_inputs.rename(None).view(batch_size, (self.num_neg + 1), self.embed_size)
        context_inputs.names = ("B", "N", "E")

        # Reshape target_inputs
        # inputs: target_inputs, shape = (B * (1 + Nneg), 1, E)
        # output: target_inputs, shape = (B, (1 + Nneg), E)
        target_inputs = target_inputs.rename(None).view(batch_size, (self.num_neg + 1), self.embed_size)
        target_inputs.names = ("B", "N", "E")

        # Index pos and neg context from context_inputs
        # inputs: context_inputs, shape = (B, (1 + Nneg), E)
        # output: context_inputs_pos, shape = (B, 1, E)
        # output: context_inputs_neg, shape = (B, Nneg, E)
        context_inputs_pos = context_inputs[:, 0, :].unflatten("E", [("N", 1), ("E", context_inputs.size("E"))])
        context_inputs_neg = context_inputs[:, 1:, :].rename(None).contiguous()
        context_inputs_neg = context_inputs_neg.view(batch_size * self.num_neg, 1, self.embed_size)
        context_inputs_neg.names = context_inputs_pos.names

        # Index pos and neg target from target_inputs
        # inputs: target_inputs, shape = (B, (1 + Nneg), E)
        # output: target_inputs_pos, shape = (B, 1, E)
        # output: target_inputs_neg, shape = (B, Nneg, E)
        target_inputs_pos = target_inputs[:, 0, :].unflatten("E", [("N", 1), ("E", target_inputs.size("E"))])
        target_inputs_neg = target_inputs[:, 1:, :].rename(None).contiguous()
        target_inputs_neg = target_inputs_neg.view(batch_size * self.num_neg, 1, self.embed_size)
        target_inputs_neg.names = target_inputs_pos.names

        # Calculate similarity between context inputs and positive samples
        # inputs: context_inputs_pos, shape = (B, 1, E)
        # inputs: target_inputs_pos, shape = (B, 1, E)
        # output: possim, shape = (B, 1)
        positive_tensor = torch.cat([context_inputs_pos, target_inputs_pos], dim="N")
        possim = self.starspace(positive_tensor)

        # Calculate similarity between context inputs and negative samples
        # inputs: context_inputs_neg, shape = (B, Nneg, E)
        # inputs: target_inputs_neg, shape = (B, Nneg, E)
        # output: possim, shape = (B, 1)
        negative_tensor = torch.cat([context_inputs_neg, target_inputs_neg], dim="N")
        negsim = self.starspace(negative_tensor)

        # Reshape possim
        # inputs: possim, shape = (B, 1)
        # output: possim, shape = (B, 1)
        possim = possim.rename(None).view(batch_size, 1)
        possim.names = ("B", "O")

        # Reshape negsim
        # inputs: negsim, shape = (B * Nneg, 1)
        # output: negsim, shape = (B, Nneg)
        negsim = negsim.rename(None).view(batch_size, self.num_neg)
        negsim.names = ("B", "O")

        # Concatenate possim and negsim on dimension O and reshape it
        # inputs: possim, shape = (B, 1)
        # inputs: negsim, shape = (B, Nneg)
        # output: outputs, shape = (B * (1 + Nneg), 1)
        outputs = torch.cat([possim, negsim], dim="O")
        outputs = outputs.rename(None).view(batch_size * (1 + self.num_neg), 1)
        outputs.names = ("B", "O")

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
