from typing import Dict

import torch
import torch.nn as nn

from torecsys.models.ltr import LtrBaseModel
from torecsys.utils.typing_extensions import Tensors


class LearningToRankWrapper(LtrBaseModel):
    """

    """

    def __init__(self, model: nn.Module):
        """

        Args:
            model:
        """
        super().__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(f'model must be nn.Module. given {type(model)} is not allowed')

        self._model = model

    def forward(self, pos_inputs: Dict[str, torch.Tensor], neg_inputs: Dict[str, torch.Tensor]) \
            -> Dict[str, Tensors]:
        """

        Args:
            pos_inputs (Dict[str, torch.Tensor]): positive inputs where the keys are the inputs arguments
                e.g. feat_inputs, emb_inputs, and the values are the tensors with the shape (B, N, E)
            neg_inputs (Dict[str, torch.Tensor]): negative inputs where the keys are the inputs arguments
                e.g. feat_inputs, emb_inputs, and the values are the tensors with the shape (B * N Neg, N, E)

        Returns:
            Dict[str, Tensors]: dictionary of outputs where pos_outputs are the tensors with shape (B, 1, _)
                and neg_outputs are the tensors with shape (B, N Neg, _)
        """
        pos_outputs = self._model(**pos_inputs)
        neg_outputs = self._model(**neg_inputs)

        # reshape neg_outputs to (B, N Neg, E)

        return {
            'pos_outputs': pos_outputs,
            'neg_outputs': neg_outputs
        }

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        Args:
            inputs:

        Returns:

        """
        return self._model(**inputs)
