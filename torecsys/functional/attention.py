import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from typing import Tuple, Union


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


def dummy_attention(key  : torch.Tensor, 
                    query: torch.Tensor, 
                    value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""dummy function for jit-compile features of torch, 
        which have the same inputs and outputs to nn.MultiheadAttention().__call__()
    
    Args:
        key (torch.Tensor): inputs to be passed as output
        query (torch.Tensor): dummy inputs
        value (torch.Tensor): dummy inputs
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: values = (key, dummy outputs = torch.Tensor([]))
    """
    return key, torch.Tensor([])
