"""torecsys.models.emb is a sub module of the implementations of Embedding algorithm"""

import torch.nn as nn

class _EmbModel(nn.Module):
    def __init__(self):
        super(_EmbModel, self).__init__()
