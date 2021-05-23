"""
torecsys.utils.operations is a sub model of utils including anything used in the package
"""

import operator as op
from functools import reduce
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn


def combination(n: int, r: int) -> int:
    """
    Calculate combination
    
    Args:
        n (int): integer of number of elements
        r (int): integer of size of combinations
    
    Returns:
        int: An integer of number of combinations
    """
    r = min(r, n - r)
    num = reduce(op.mul, range(n, n - r, -1), 1)
    den = reduce(op.mul, range(1, r + 1), 1)
    return int(num / den)


def dummy_attention(key: torch.Tensor,
                    query: torch.Tensor,
                    value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dummy tensor which have the same inputs and outputs to nn.MultiHeadAttention().__call__()
    
    Args:
        key (T): inputs to be passed as output
        query (T): dummy inputs
        value (T): dummy inputs
    
    Returns:
        Tuple[T, T]: values = (key, dummy outputs = torch.Tensor([]))
    """
    return key, torch.Tensor([])


def inner_product_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Calculate inner product of two vectors
    
    Args:
        a (T, shape = (B, N_{a}, E)), data_type=torch.float: the first batch of vector to be multiplied
        b (T, shape = (B, N_{b}, E)), data_type=torch.float: the second batch of vector to be multiplied
        dim (int): dimension to sum the tensor
    
    Returns:
        T, data_type=torch.float: inner product tensor
    """
    return (a * b).sum(dim=dim)


def regularize(parameters: List[Tuple[str, nn.Parameter]],
               weight_decay: float = 0.01,
               norm: int = 2) -> torch.Tensor:
    """
    Calculate p-th order regularization
    
    Args:
        parameters (List[Tuple[str, nn.Parameter]]): parameters to calculate the regularized loss
        weight_decay (float, optional): multiplier of regularized loss. Defaults to 0.01
        norm (int, optional): order of norm to calculate regularized loss. Defaults to 2
    
    Returns:
        T, data_type=torch.float: regularized loss
    """
    loss = 0.0

    for name, param in parameters:
        if 'weight' in name:
            loss += torch.norm(param, p=norm)

    return torch.Tensor([loss * weight_decay]).data[0]


def replicate_tensor(tensor: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    """
    Replicate tensor by batch / by row
    
    Args:
        tensor (T), shape = (B, ...): tensor to be replicated
        size (int): size to replicate tensor
        dim (int): dimension to replicate tensor. Replicated by batch if dim = 0, else by row
    
    Returns:
        T, shape = (B * size, ...): replicated Tensor
    """
    # Get shape of tensor from pos_samples
    # inputs: tensor, shape = (B, ...)
    # output: batch_size, int, values = B
    # output: tensor_shape, tuple, values = (...)
    batch_size = tensor.size()[0]
    tensor_shape = tuple(tensor.size()[1:])

    # Unsqueeze by dim 1 and repeat n-times by dim 1
    # inputs: tensor, shape = (B, ...)
    # output: repeat_tensor, shape = (B, size, ...) / (1, B * size, ...)
    repeat_size = (1, size) + tuple([1 for _ in range(len(tensor_shape))])
    repeat_tensor = tensor.unsqueeze(dim).repeat(repeat_size)

    # Reshape repeat_tensor to (batch_size * size, ...)
    # inputs: repeat_tensor, shape = (B, size, ...) / (1, B * size, ...)
    # output: repeat_tensor, shape = (B * size, ...)
    reshape_size = (batch_size * size,) + tensor_shape
    return repeat_tensor.view(reshape_size)


def show_attention(attentions: np.ndarray,
                   x_axis: Optional[Union[list, str]] = None,
                   y_axis: Optional[Union[list, str]] = None,
                   save_dir: Optional[str] = None):
    """
    Show attention of MultiHeadAttention in a mpl heatmap
    
    Args:
        attentions (np.ndarray), shape = (L, L), data_type=np.float32: attentions weights of nn.MultiHeadAttention
        x_axis (str, optional): string or list of x axis. Defaults to None
        y_axis (str, optional): string or list of y axis. Defaults to None
        save_dir (str, optional): string of directory to save the attention png. Defaults to None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions)
    fig.colorbar(cax)

    if x_axis is not None:
        if isinstance(x_axis, str):
            x_axis = [''] + x_axis.split(',')
        elif isinstance(x_axis, list):
            x_axis = [''] + x_axis
        else:
            raise TypeError(f'type of x_axis {type(x_axis)} is not acceptable')

        ax.set_xticklabels(x_axis, rotation=90)

    if y_axis is not None:
        if isinstance(y_axis, str):
            y_axis = [''] + y_axis.split(',')
        elif isinstance(y_axis, list):
            y_axis = [''] + y_axis
        else:
            raise TypeError(f'type of y_axis {type(y_axis)} is not acceptable')

        ax.set_yticklabels(y_axis)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show() if save_dir is None else plt.savefig(save_dir)


def squash(inputs: torch.Tensor, dim: Optional[int] = -1) -> torch.Tensor:
    """
    Apply `squash` non-linearity to inputs
    
    Args:
        inputs (T): input tensor which is to be applied squashing
        dim (int, optional): dimension to be applied squashing. Defaults to -1
    
    Returns:
        T: squashed tensor
    """
    # Calculate squared norm of inputs
    squared_norm = torch.sum(torch.pow(inputs, 2), dim=dim, keepdim=True)

    # Squash non-linearity to inputs
    return (squared_norm / (1 + squared_norm)) * (inputs / (torch.sqrt(squared_norm) + 1e-8))
