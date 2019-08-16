from . import _NegativeSampler
import torch


class MultinomialSampler(_NegativeSampler):
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
