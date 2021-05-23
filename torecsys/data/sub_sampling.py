from typing import Union

import numpy as np
import pandas as pd


def sub_sampling(data: Union[np.ndarray, pd.DataFrame],
                 key: Union[int, str],
                 formula: str = "paper",
                 threshold: float = 1e-5) -> Union[np.ndarray, pd.DataFrame]:
    r"""
    Drop occurrences of the most frequent word tokens with the following condition by the given threshold
    value: if :math:`P_{\text{random}} > P_{\text{drop}}`, where :math:`P_{\text{random}} \in U(0, 1)` and 
    :math:`P_{drop} = 1 - \sqrt{\frac{t}{f}}` or :math:`P_{drop} = \frac{f - t}{f} - \sqrt{\frac{t}{f}}`,
    where :math:`f = frequency and t = threshold.`

    Hence,

    #. more samples of token will be dropped if the word token is more frequent, when threshold is larger.

    #. samples of word token where the frequent of it is lower than the threshold will not be dropped.
    
    Args:
        data (Union[np.ndarray, pd.DataFrame], shape = (Number of samples, ...)): A np.array or pd.DataFrame of
            raw data.
        key (Union[int, str]): An integer of index or a string of name of target field to be used for sub_sampling.
        formula (str): A string to select which formula to calculate drop ratio in sub_sampling.
            Allows: [\"code\"|\"paper\"].
        threshold (float, optional): A float of threshold of sub_sampling.
            Defaults to 1e-5.
    
    Raises:
        ValueError: when formula is not in [\"code\"|\"paper\"].
    
    Returns:
        Union[np.ndarray, pd.DataFrame], shape = (Number of subsample samples, ...): A np.array or pd.DataFrame of
            subsample data.

    :Reference:

    #. `Tomas Mikolov et al, 2013. Distributed Representations of Words and Phrases and their Compositional
        <https://arxiv.org/abs/1310.4546>`_.
    
    #. `Omer Levy et al, 2015. Improving Distributional Similarity with Lessons Learned from Word Embeddings
        <https://levyomer.files.wordpress.com/2015/03/improving-distributional-similarity-tacl-2015.pdf>`_.
    """
    if formula == 'code':
        def subsampling_prob(f, t):
            return (f - t) / f - np.sqrt(t / f)
    elif formula == 'paper':
        def subsampling_prob(f, t):
            return 1 - np.sqrt(t / f)
    else:
        raise ValueError('formula only allows ["code", "paper"].')

    if isinstance(data, np.ndarray):
        def get_row(d, row_key):
            return d[row_key]

        def get_col(d, col_key):
            return d[:, col_key]

        def get_value(d, row_key, col_key):
            return d[row_key, col_key]

        def concat(samp):
            return np.vstack(samp)

    elif isinstance(data, pd.DataFrame):
        def get_row(d, row_key):
            return d.iloc[row_key]

        def concat(samp):
            return pd.DataFrame(samp, columns=data.columns).reset_index(drop=True)

        if isinstance(key, int):
            def get_col(d, col_key):
                return d.iloc[:, col_key]

            def get_value(d, row_key, col_key):
                return d.iloc[row_key, col_key]

        else:
            def get_col(d, col_key):
                return d[col_key]

            def get_value(d, row_key, col_key):
                return d.loc[row_key, col_key]
    else:
        raise TypeError(f'type of data {data.__class__} is not allowed, only allow [np.ndarray, pd.DataFrame].')

    columns = get_col(data, key)
    uniques, counts = np.unique(columns, return_counts=True)
    freq = counts / np.sum(counts)
    prob = subsampling_prob(freq, threshold)
    prob = dict(zip(uniques, prob))
    rand_prob = np.random.random(size=(data.shape[0]))
    sampled_data = []

    for i in range(data.shape[0]):
        if rand_prob[i] > prob[get_value(data, i, key)]:
            sampled_data.append(get_row(data, i))

    return concat(sampled_data)
