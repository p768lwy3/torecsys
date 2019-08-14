import numpy as np

def subsampling(array     : np.ndarray, 
                column_id : int   = 0, 
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
    # count occurences of each token
    uniques, counts = np.unique(array[:, column_id], return_counts=True)
    
    # calculate the frequencies of each token
    frequencies = counts / np.sum(counts)
    
    # calculate the subsampling probabilities
    probabilities = 1 - np.sqrt(threshold / frequencies)
    
    # created subsampling probabilities dict
    probabilities = dict(zip(uniques, probabilities))
    
    # generate random values for each row of samples
    random_probabilities = np.random.random(size=(array.shape[0]))
    
    # subsampling
    subsampled_data = []
    for i in range(array.shape[0]):
        if random_probabilities[i] < probabilities[array[i, column_id]]:
            subsampled_data.append(array[i])
    
    return np.vstack(subsampled_data)
    