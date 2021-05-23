"""
torecsys.embedder is a sub-model to transform or embed embedder to Tensor.
"""

__all__ = [
    'Inputs',
    'ConcatInput',
    'ImageInput',
    'BaseInput',
    'ListIndicesEmbedding',
    'MultiIndicesEmbedding',
    'MultiIndicesFieldAwareEmbedding',
    'PretrainedImageInput',
    'SequenceIndicesEmbedding',
    'SingleIndexEmbedding',
    'StackedInput',
    'ValueInput'
]

from torecsys.inputs.base import *
from torecsys.inputs.inputs import Inputs
