r"""shorcut to call torecsys.models.*.losses.* classes and functions
"""

class _Loss(nn.Module):
    def __init__(self):
        super(_Loss, self).__init__()
