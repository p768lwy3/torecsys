from . import _EmbModel
from torecsys.layers import StarSpaceLayer
from torecsys.functional import inner_product_similarity
from torecsys.utils.decorator import jit_experimental
from functools import partial
import torch
from typing import Callable, Tuple


class StarSpaceModel(_EmbModel):
    r"""StatSpaceModel"""
    def __init__(self, 
                 embed_size: int,
                 num_neg   : int,
                 similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = partial(inner_product_similarity, dim=2)):
        r"""[summary]
        
        Args:
            similarity (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): function to calculate the similarity between two vectors. Defaults to partial(inner_product_similarity, dim=2).
        """
        super(StarSpaceModel, self).__init__()
        self.embed_size = embed_size
        self.num_neg = num_neg
        self.starspace = StarSpaceLayer(similarity)

    def forward(self, 
                context_inputs: torch.Tensor, 
                target_inputs : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""feed forward of starspace
        
        Args:
            context_inputs (T), shape = (B * (1 + Nneg), 1, E), dtype = torch.float: embedding tensors of context items
            target_inputs (T), shape = (B * (1 + Nneg), 1, E), dtype = torch.float: embedding tensors of target items
        
        Returns:
            Tuple[T, T], shape = (B, 1 / Nneg): similarity scores between context and positive or negative samples
        """
        # reshape inputs into (B, (1+ Nneg), E)
        agg_batch_size = context_inputs.size(0)
        batch_size = int(agg_batch_size // (1 + self.num_neg))

        context_inputs = context_inputs.view(batch_size, (self.num_neg + 1), self.embed_size)
        target_inputs = target_inputs.view(batch_size, (self.num_neg + 1), self.embed_size)

        # index pos / neg context (target) from context_inputs (target_inputs)
        context_inputs_pos = context_inputs[:, 0, :].unsqueeze(1)
        context_inputs_neg = context_inputs[:, 1:, :].contiguous().view(batch_size * self.num_neg, 1, self.embed_size)

        target_inputs_pos = target_inputs[:, 0, :].unsqueeze(1)
        target_inputs_neg = target_inputs[:, 1:, :].contiguous().view(batch_size * self.num_neg, 1, self.embed_size)

        # calculate similarity between context inputs and positive samples
        positive_tensor = torch.cat([context_inputs_pos, target_inputs_pos], dim=1)
        possim = self.starspace(positive_tensor)

        # calculate similarity between context inputs and negative samples
        negative_tensor = torch.cat([context_inputs_neg, target_inputs_neg], dim=1)
        negsim = self.starspace(negative_tensor)

        # stack the output of pos sim and neg sim into (B * (1+ neg), 1)
        possim = possim.view(batch_size, 1)
        negsim = negsim.view(batch_size, self.num_neg)
        outputs = torch.cat([possim, negsim], dim=1).view(batch_size * (1 + self.num_neg), 1)
        
        return outputs
        