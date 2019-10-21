import torch

def inner_product_similarity(a: torch.Tensor, b: torch.Tensor, dim=1) -> torch.Tensor:
    r"""calculate inner-product of two vectors
    
    Args:
        a (T, shape = (B, N_{a}, E)), dtype = torch.float: the first batch of vector to be multiplied.
        b (T, shape = (B, N_{b}, E)), dtype = torch.float: the second batch of vector to be multiplied.
    
    Returns:
        T, dtype = torch.float: inner product tensor.
    """
    outputs = (a * b).sum(dim=dim)
    return outputs
