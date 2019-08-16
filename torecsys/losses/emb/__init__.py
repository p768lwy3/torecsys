r"""torecsys.losses.emb is a sub module of loss functions used in Embedding
"""

from .. import _Loss

class _EmbLoss(_Loss):
    def __init__(self):
        super(_EmbLoss, self).__init__()

# from .functional import *
# from .skipgram import SkipGramLoss
