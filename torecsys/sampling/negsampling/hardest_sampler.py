from typing import Dict

import torch

from . import _NegativeSampler


class HardestNegativeSampler(_NegativeSampler):
    r"""HardestNegativeSampler is to generate the hardest negative sampler, 
    i.e. the negatives closest to each training query.

    :Reference:
    
    #. `Fartash Faghri, 2017. VSE++: Improving Visual-Semantic Embeddings with Hard Negatives
    <https://arxiv.org/abs/1707.05612>`_.

    """

    def size(self) -> Dict[str, int]:
        r"""Get length of field.
        
        Returns:
            Dict[str, int]: Length of field.
        """
        return {k: 1 for k, _ in self.kwargs.items()}

    def _generate(self) -> torch.Tensor:
        r"""A function to generate the hardest negative samples within a batch.

        Returns:
            T, shape = (N * 1, 1), dtype = torch.long: Tensor of the hardest negative samples of queries.
        """
        # For each row (query) of batch, 

        # Generate negative samples of the query by batch excluding itself

        # Calculate scores of negative samples with the model

        # Calculate the sorting order of the scores in descending order

        # Return the hardest negative sample of the given positive sample

        return None
