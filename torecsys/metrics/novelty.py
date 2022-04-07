from typing import Dict, Any

import numpy as np
import torch

from torecsys.metrics import BaseMetric


class Novelty(BaseMetric):
    """
    Compute the novelty for a list of recommendations

    Metric Definition:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """

    def __init__(self, occurrence: Dict[int, int], k: int, num_users: int, dist_sync_on_step=False):
        """

        Args:
            occurrence:
            k:
            num_users:
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.occurrence = occurrence
        self.k = k
        self.num_users = num_users

        self.add_state('predictions', [], dist_reduce_fx='cat')
        self.add_state('mean_self_information', [], dist_reduce_fx='cat')

    def update(self, predictions: torch.Tensor, **__: Any) -> None:
        self.predictions.append(predictions)

    def compute(self) -> float:
        predictions = torch.hstack(self.predictions)
        num_predictions = len(self.predictions)

        for p in predictions:
            self_information = 0
            for el in p:
                self_information += np.sum(-np.log2(self.occurrence[el.item()] / self.num_users))

            self.mean_self_information.append(self_information / self.k)

        novelty = sum(self.mean_self_information) / num_predictions
        return novelty
