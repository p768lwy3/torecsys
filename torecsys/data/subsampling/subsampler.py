import numpy as np
import pandas as pd
from typing import Union

def subsampling(data      : Union[np.ndarray, pd.DataFrame], 
                key       : Union[int, str], 
                threshold : float = 1e-2) -> np.ndarray:
    r"""Drop occurrences of the most frequent word tokens with the following condition by the given threshold value:
    if :math:`P_{\text{random}} < P_{\text{drop}}`, where :math:`P_{\text{random}} \in U(0, 1)` and :math:`P_{drop} = 1 - \sqrt{\frac{threshold}{frequencies}}`

    Hence

    #. more samples of token will be droped if the word token is more frequent, when threshold is larger.

    #. samples of word token where the frequent of it is lower than the threshold will not be dropped.
    
    Args:
        array (np.ndarray, shape = (Number of samples, ...)): dataset to be sampled.
        column_id (int, optional): column index of the target field to be subsampled. Defaults to 0.
        threshold (float, optional): threshold values of sub-sampling. Defaults to 1e-2.
    
    Returns:
        np.ndarray, shape = (Number of subsampled samples, ...): subsampled dataset

    :Reference:

    #. `Omer Levy et al, 2015. Improving Distributional Similarity with Lessons Learned from Word Embeddings <https://levyomer.files.wordpress.com/2015/03/improving-distributional-similarity-tacl-2015.pdf>`_.

    """
    # initialize columns' selector with lambda function
    if isinstance(data, np.ndarray):
        row_func = lambda data, row_key: data[row_key]
        col_func = lambda data, col_key: data[:, col_key]
        val_func = lambda data, row_key, col_key: data[row_key, col_key]
    elif isinstance(data, pd.DataFrame):
        row_func = lambda data, row_key: data.iloc[row_key]
        
        if isinstance(key, int):
            col_func = lambda data, col_key: data.iloc[:, col_key]
            val_func = lambda data, row_key, col_key: data.iloc[row_key, col_key]
        else:
            col_func = lambda data, col_key: data[col_key]
            val_func = lambda data, row_key, col_key: data.loc[row_key, col_key]

    # select columns from data
    columns = col_func(data, key)
    
    # count occurences of each token
    uniques, counts = np.unique(columns, return_counts=True)
    
    # calculate the frequencies of each token
    freq = counts / np.sum(counts)
    
    # calculate the subsampling probabilities
    prob = 1 - np.sqrt(threshold / freq)
    
    # created subsampling probabilities dict
    prob = dict(zip(uniques, prob))
    
    # generate random values for each row of samples
    rand_prob = np.random.random(size=(data.shape[0]))
    
    # subsampling
    subsampled_data = []
    for i in range(data.shape[0]):
        if rand_prob[i] < prob[val_func(data, i, key)]:
            subsampled_data.append(row_func(data, i))
    
    return np.vstack(subsampled_data)
