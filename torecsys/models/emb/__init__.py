r"""torecsys.models.ctr is a sub module of the implementation of the whole models of Embedding model
"""

from .. import _Model


class _EmbModel(_Model):
    def __init__(self):
        super(_EmbModel, self).__init__()


from .matrix_factorization import MatrixFactorizationModel
from .starspace import StarSpaceModel
