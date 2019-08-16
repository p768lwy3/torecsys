r"""torecsys.data.negsampling is a module of negative sampling algorithms, e.g. MultinomialSampler, UniformSampler
"""

class _NegativeSampler(object):
    r"""Base Class of Negative Sampler
    """
    def __init__(self):
        raise NotImplementedError("")

    def __len__(self) -> int:
        r"""Return size of dictionary.
        
        Returns:
            int: total number of words in dictionary
        """
        raise self.dict_size

    def __call__(self, size: int) -> torch.Tensor:
        return self.generate(size)

    def generate(self, size: int) -> torch.Tensor:
        r"""Return drawn samples.
        
        Args:
            size (int): Number of negative samples to be drawn
        
        Raises:
            NotImplementedError: not implementated in base class
        
        Returns:
            torch.Tensor, shape = (size, 1), dtype = torch.long: Drawn negative samples
        """
        raise NotImplementedError("")

from .multinomial_sampler import MultinomialSampler
from .uniform_sampler import UniformSamplerWithoutReplacement
from .uniform_sampler import UniformSamplerWithReplacement
