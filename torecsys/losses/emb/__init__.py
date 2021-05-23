"""
torecsys.losses.emb is a sub model of implementation of losses in embedding.
"""

from torecsys.losses import Loss


class EmbLoss(Loss):
    """
    General Embedding Loss class.
    """

    def __init__(self):
        super().__init__()


__all__ = [
    'EmbLoss',
    'SkipGramLoss'
]

from torecsys.losses.emb.skipgram import SkipGramLoss
