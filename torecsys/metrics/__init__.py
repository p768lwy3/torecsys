"""
torecsys.metrics is a sub model to implement functions to calculate metrics of evaluation in recommendation system.
"""
from typing import Any

import torchmetrics

from torecsys.metrics import *


class BaseMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def update(self, *_: Any, **__: Any) -> None:
        pass

    def compute(self) -> Any:
        pass

from torecsys.metrics.novelty import Novelty