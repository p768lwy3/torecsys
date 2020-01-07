r"""torecsys.utils.opeations is a sub module of utils including anything used in the package, 
and I don't know where should I put them to.
"""
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import operator as op
import torch
import torch.nn as nn
from typing import List, Tuple, Tuple, Union

def combination(n: int, r: int) -> int:
    r"""function to calculate combination.
    
    Args:
        n (int): An integer of number of elements
        r (int): An integer of size of combinations
    
    Returns:
        int: An integer of number of combinations.
    """
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return int(numer / denom)

def dummy_attention(key  : torch.Tensor, 
                    query: torch.Tensor, 
                    value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""function for dummy in jit-compile features of torch, which have the same inputs and 
    outputs to nn.MultiheadAttention().__call__()
    
    Args:
        key (T): inputs to be passed as output
        query (T): dummy inputs
        value (T): dummy inputs
    
    Returns:
        Tuple[T, T]: values = (key, dummy outputs = torch.Tensor([]))
    """
    return key, torch.Tensor([])

def inner_product_similarity(a: torch.Tensor, b: torch.Tensor, dim=1) -> torch.Tensor:
    r"""function to calculate inner-product of two vectors
    
    Args:
        a (T, shape = (B, N_{a}, E)), dtype = torch.float: the first batch of vector to be multiplied.
        b (T, shape = (B, N_{b}, E)), dtype = torch.float: the second batch of vector to be multiplied.
    
    Returns:
        T, dtype = torch.float: inner product tensor.
    """
    outputs = (a * b).sum(dim=dim)
    return outputs

def regularize(parametes    : List[Tuple[str, nn.Parameter]], 
               weight_decay : float = 0.01,
               norm         : int = 2) -> torch.Tensor:
    r"""function to calculate p-th order regularization of paramters in the model
    
    Args:
        parametes (List[Tuple[str, nn.Parameter]]): list of tuple of names and paramters to 
            calculate the regularized loss
        weight_decay (float, optional): multiplier of regularized loss. Defaults to 0.01.
        norm (int, optional): order of norm to calculate regularized loss. Defaults to 2.
    
    Returns:
        T, shape = (1, ), dtype = torch.float: regularized loss
    """
    loss = 0.0
    for name, param in parametes:
        if "weight" in name:
            loss += torch.norm(param, p=norm)
    
    return loss * weight_decay

def show_attention(attentions : np.ndarray, 
                   xaxis      : Union[list, str] = None, 
                   yaxis      : Union[list, str] = None, 
                   savedir    : str = None):
    r"""Show attention of MultiheadAttention in a mpl heatmap
    
    Args:
        attentions (np.ndarray), shape = (sequence length, sequence length), dtype = np.float32: Attentions Weights of output of nn.MultiheadAttention
        xaxis (str, optional): string or list of xaxis. Defaults to None.
        yaxis (str, optional): string or list of yaxis. Defaults to None.
        savedir (str, optional): string of directory to save the attention png. Defaults to None.
    """
    # set up figure with colorbar
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    cax = ax.matshow(attentions)
    fig.colorbar(cax)

    # set up axes
    if xaxis is not None:
        if isinstance(xaxis, str):
            xaxis = [""] + xaxis.split(",")
        elif isinstance(xaxis, list):
            xaxis = [""] + xaxis
        ax.set_xticklabels(xaxis, rotation=90)
    
    if yaxis is not None:
        if isinstance(yaxis, str):
            yaxis = [""] + yaxis.split(",")
        elif isinstance(yaxis, list):
            yaxis = [""] + yaxis
        ax.set_yticklabels(yaxis)
    
    # show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    if savedir is None:
        plt.show()
    else:
        plt.savefig(savedir)

def squash(inputs: torch.Tensor, dim=-1) -> torch.Tensor:
    r"""apply `squashing` non-linearity to inputs
    
    Args:
        inputs (T): Inputs tensor which is to be applied squashing.
        dim (int, optional): Dimension to be applied squashing. 
            Defaults to -1.
    
    Returns:
        T: Squashed tensor.
    """
    # calculate squared norm of inputs
    squared_norm = torch.sum(torch.pow(inputs, 2), dim=dim, keepdim=True)

    # apply `squashing` non-linearity to inputs
    c_j = (squared_norm / (1 + squared_norm)) * (inputs / (torch.sqrt(squared_norm) + 1e-8))

    return c_j
    