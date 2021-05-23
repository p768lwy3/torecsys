from functools import partial
from typing import Any

import torch

from torecsys.layers import StarSpaceLayer
from torecsys.models.emb import EmbBaseModel
from torecsys.utils.operations import inner_product_similarity


class StarSpaceModel(EmbBaseModel):
    """
    Model class of StatSpaceModel.

    Starspace is a model proposed by Facebook in 2017, which to embed relationship between different kinds of pairs,
    like (word, tag), (user-group) etc, by calculating similarity between context and positive samples or negative
    samples, and margin ranking loss between similarities.

    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`_.
    
    """

    def __init__(self,
                 embed_size: int,
                 num_neg: int,
                 similarity: Any = partial(inner_product_similarity, dim=2)):
        """
        Initialize StarSpaceModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_neg (int): number of negative samples
            similarity (Any, optional): function of similarity between two tensors.
                e.g. torch.nn.functional.cosine_similarity. Defaults to partial(inner_product_similarity, dim=2)
        """
        super().__init__()

        self.embed_size = embed_size
        self.num_neg = num_neg
        self.starspace = StarSpaceLayer(similarity)

    def forward(self,
                context_inputs: torch.Tensor,
                target_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of StarSpaceModel
        
        Args:
            context_inputs (T), shape = (B * (1 + N Neg), 1, E), data_type = torch.float:
                embedded features tensors of context
            target_inputs (T), shape = (B * (1 + N Neg), 1, E), data_type = torch.float:
                embedded features tensors of target
        
        Returns:
            T, shape = (B, 1 / N Neg): Output of StarSpaceModel.
        """
        # Name the feat_inputs tensor for flatten
        context_inputs.names = ('B', 'N', 'E',)

        # Get batch size from context_inputs
        agg_batch_size = context_inputs.size('B')
        batch_size = int(agg_batch_size // (1 + self.num_neg))

        # Reshape context_inputs 
        # inputs: context_inputs, shape = (B * (1 + N Neg), 1, E)
        # output: context_inputs, shape = (B, (1 + N Neg), E)
        context_inputs = context_inputs.rename(None).view(batch_size, (self.num_neg + 1), self.embed_size)
        context_inputs.names = ('B', 'N', 'E',)

        # Reshape target_inputs
        # inputs: target_inputs, shape = (B * (1 + N Neg), 1, E)
        # output: target_inputs, shape = (B, (1 + N Neg), E)
        target_inputs = target_inputs.rename(None).view(batch_size, (self.num_neg + 1), self.embed_size)
        target_inputs.names = ('B', 'N', 'E',)

        # Index pos and neg context from context_inputs
        # inputs: context_inputs, shape = (B, (1 + N Neg), E)
        # output: context_inputs_pos, shape = (B, 1, E)
        # output: context_inputs_neg, shape = (B, N Neg, E)
        context_inputs_pos = context_inputs[:, 0, :].unflatten('E', (('N', 1,), ('E', context_inputs.size('E'),),))
        context_inputs_neg = context_inputs[:, 1:, :].rename(None).contiguous()
        context_inputs_neg = context_inputs_neg.view(batch_size * self.num_neg, 1, self.embed_size)
        context_inputs_neg.names = context_inputs_pos.names

        # Index pos and neg target from target_inputs
        # inputs: target_inputs, shape = (B, (1 + N Neg), E)
        # output: target_inputs_pos, shape = (B, 1, E)
        # output: target_inputs_neg, shape = (B, N Neg, E)
        target_inputs_pos = target_inputs[:, 0, :].unflatten('E', (('N', 1,), ('E', context_inputs.size('E'),),))
        target_inputs_neg = target_inputs[:, 1:, :].rename(None).contiguous()
        target_inputs_neg = target_inputs_neg.view(batch_size * self.num_neg, 1, self.embed_size)
        target_inputs_neg.names = target_inputs_pos.names

        # Calculate similarity between context inputs and positive samples
        # inputs: context_inputs_pos, shape = (B, 1, E)
        # inputs: target_inputs_pos, shape = (B, 1, E)
        # output: pos_sim, shape = (B, 1)
        positive_tensor = torch.cat([context_inputs_pos, target_inputs_pos], dim='N')
        pos_sim = self.starspace(positive_tensor)

        # Calculate similarity between context inputs and negative samples
        # inputs: context_inputs_neg, shape = (B, N Neg, E)
        # inputs: target_inputs_neg, shape = (B, N Neg, E)
        # output: pos_sim, shape = (B, 1)
        negative_tensor = torch.cat([context_inputs_neg, target_inputs_neg], dim="N")
        neg_sim = self.starspace(negative_tensor)

        # Reshape pos_sim
        # inputs: pos_sim, shape = (B, 1)
        # output: pos_sim, shape = (B, 1)
        pos_sim = pos_sim.rename(None).view(batch_size, 1)
        pos_sim.names = ('B', 'O',)

        # Reshape neg_sim
        # inputs: neg_sim, shape = (B * N Neg, 1)
        # output: neg_sim, shape = (B, N Neg)
        neg_sim = neg_sim.rename(None).view(batch_size, self.num_neg)
        neg_sim.names = ('B', 'O',)

        # Concatenate pos_sim and neg_sim on dimension O and reshape it
        # inputs: pos_sim, shape = (B, 1)
        # inputs: neg_sim, shape = (B, N Neg)
        # output: outputs, shape = (B * (1 + N Neg), 1)
        outputs = torch.cat([pos_sim, neg_sim], dim='O')
        outputs = outputs.rename(None).view(batch_size * (1 + self.num_neg), 1)
        outputs.names = ('B', 'O',)

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs

    def predict(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented
