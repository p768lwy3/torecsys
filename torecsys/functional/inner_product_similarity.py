import torch

def inner_product_similarity(a: torch.Tensor, b: torch.Tensor, dim=1) -> torch.Tensor:
    r"""calculate inner-product of two vectors
    
    Args:
        a (torch.Tensor): input vector
        b (torch.Tensor): input vector
        dim (int, optional): aggregation dimension. Defaults to 1.
    
    Returns:
        torch.Tensor: inner product tensor, shape = :math:`(x_{0}, x_{1}, ..., x_{i-1}, x_{i+1}, ...) \text{, where i is the input dim.}` 
    """
    outputs = (a * b).sum(dim=dim)
    return outputs
