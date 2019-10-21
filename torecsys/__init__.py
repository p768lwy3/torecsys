r"""ToR[e]cSys is a package implemented different kinds of recommendation system algorithm in PyTorch!
"""

__version__ = "0.0.1-dev"

import torecsys.data
import torecsys.estimators
import torecsys.functional
import torecsys.inputs
import torecsys.layers
import torecsys.losses
import torecsys.metrics
import torecsys.models
import torecsys.utils

from .utils.training.ranking_trainer import RankingTrainer
from .utils.training.trainer import Trainer

