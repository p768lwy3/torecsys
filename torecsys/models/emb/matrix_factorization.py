from . import _EmbModule
from ..layers import GeneralizedMatrixFactorizationLayer
from torecsys.utils.logging.decorator import jit_experimental
import torch
from typing import Dict

class MatrixFactorizationModule(_EmbModule):
    r""""""
    def __init__(self):
        super(MatrixFactorizationModule, self).__init__()
    
    def 