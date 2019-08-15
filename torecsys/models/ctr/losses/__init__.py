r"""torecsys.models.ctr.losses is a sub module of loss functions used in Click Through Rate Prediction
"""

from torecsys.losses import _Loss

class _CtrLoss(_Loss):
    def __init__(self):
        super(_CtrLoss, self).__init__()
