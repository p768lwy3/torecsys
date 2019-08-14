import torch

class BaseNegativeSampler(object):
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

class MultinomialSampler(BaseNegativeSampler):
    r"""MutlinomialSampler is to generate negative samplers by multinomial distribution, i.e. draw samples by given probabilities
    """
    def __init__(self, 
                 weights          : torch.Tensor,
                 with_replacement : bool = True):
        r"""Initialize a Negative sampler which draw samples with multinomial distribution
        
        Args:
            weights (torch.Tensor): weights (probabilities) to draw samples, with shape = (total number of words in dictionary, ).
            with_replacement (bool, optional): boolean flag to control the replacement of sampling. Defaults to True.
        """
        self.with_replacement = with_replacement
        if isinstance(weights, torch.Tensor):
            self.weights = weights
        else:
            self.weights = torch.Tensor(weights)
        self.dict_size = len(self.weights)
        
    def generate(self, size: int) -> torch.Tensor:
        r"""Return drawn samples.
        
        Args:
            size (int): Number of negative samples to be drawn
        
        Returns:
            torch.Tensor, shape = (size, 1), dtype = torch.long: Drawn negative samples
        """
        samples = torch.multinomial(self.weights, size, replacement=self.with_replacement)
        return samples.long()


class UniformSamplerWithReplacement(BaseNegativeSampler):
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


class UniformSamplerWithoutReplacement(BaseNegativeSampler):
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
