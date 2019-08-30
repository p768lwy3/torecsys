r"""torecsys.losses.emb is a sub module of implementation of losses in embedding.
"""

from .. import _Loss

class _EmbLoss(_Loss):
    r"""General Embedding Loss class.
    """
    def __init__(self):
        # refer to parent class
        super(_EmbLoss, self).__init__()

from .functional import *
from .skipgram import SkipGramLoss
