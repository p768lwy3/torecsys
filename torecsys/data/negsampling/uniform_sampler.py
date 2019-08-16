from . import _NegativeSampler
import torch


class UniformSamplerWithReplacement(_NegativeSampler):
    r"""UniformSamplerWithReplacement is to generate negative samplers by uniform distribution with replacement, i.e. draw samples uniformlly with replacement
    """
    def __init__(self, 
                 low  : int, 
                 high : int):
        r"""Initialize a Negative sampler which draw samples with uniform distribution with replacement
        
        Args:
            low (int): minimum value (i.e. lower bound) of sampling id.
            high (int): maximum value (i.e. upper bound) of sampling id.
        """
        self.low = low
        self.high = high
        self.dict_size = self.high - self.low
        
    def generate(self, size: int) -> torch.Tensor:
        r"""Return drawn samples.
        
        Args:
            size (int): Number of negative samples to be drawn
        
        Returns:
            torch.Tensor, shape = (size, 1), dtype = torch.long: Drawn negative samples
        """
        return torch.randint(low=self.low, high=self.high, size=(size, )).long()


class UniformSamplerWithoutReplacement(_NegativeSampler):
    r"""UniformSamplerWithReplacement is to generate negative samplers by uniform distribution without replacement, i.e. draw samples uniformlly without replacement
    """
    def __init__(self, 
                 low  : int, 
                 high : int):
        r"""Initialize a Negative sampler which draw samples with uniform distribution without replacement

        Args:
            low (int): minimum value (i.e. lower bound) of sampling id.
            high (int): maximum value (i.e. upper bound) of sampling id.
        """
        self.low = low
        self.high = high
        self.dict_size = self.high - self.low
        
    def generate(self, size: int) -> torch.Tensor:
        r"""Generate negative samples by the sampler
        
        Args:
            size (int): Number of negative samples to be drawn
        
        Raises:
            ValueError: if input size is larger than the size of dictionary (i.e. high - low)
        
        Returns:
            torch.Tensor, shape = (size, 1), dtype = torch.long: Drawn negative samples
        """
        
        if size >= (self.high - self.low):
            raise ValueError("input size cannot be larger than size of samples.")
        
        samples = torch.randperm(n=self.high) + self.low
        samples = samples[:size]
        return samples.long()
