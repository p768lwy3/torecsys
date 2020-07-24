import torch


def mean_average_precision_at_k(act: torch.Tensor,
                                pred: torch.Tensor,
                                k: int = 10) -> torch.Tensor:
    r"""Calculate the mean of average precision at k between two tensors of batch of items, 
    i.e. `torch.tensor([[1, 2, 3], [2, 1, 4], ...])` and `torch.tensor([[1, 2, 4, 5], [2, 4, 3, 6], ...])`.
    
    Args:
        act (T), shape = (B, O), dtype = torch.int: A tensor of list of actual items to be predicted
        pred (T), shape = (B, O), dtype = torch.int: A tensor of list of predicted items
        k (int, optional): The maximum number of predicted elements. Defaults to 10.
    
    Returns:
        T: The mean of average precision at k over the batch of items.
    """
    # check if k is smaller than number of outputs's items
    k = pred.size("O") if k > pred.size("O") else k

    # get batch size
    b = pred.size("B")

    # create zeros-like tensor to store values
    scores = torch.zeros(size=(b, 1))
    num_hits = torch.zeros(size=(b, 1))

    # loop through k
    for i in range(k):
        # get pred[:, i] and reshape back to (B, O)
        p = pred[:, i].unflatten("B", [("B", b), ("O", 1)])
        # if pred[row, i] not in pred[row, :i], return True, else False
        not_p = ~(p == pred[:, :i]).sum(dim="O", keepdim=True).bool()
        # if pred[row, i] in act[row, :], return True, else False
        hits = (p == act).sum(dim="O", keepdim=True).bool()
        # sum not_p and hit and convert to integer
        hits = (hits.rename(None) & not_p.rename(None)).int()

        # add hit to score and num_hits
        num_hits += hits
        score = num_hits.masked_fill(~(hits.bool()), 0) / (i + 1)
        scores += score

    ## scores.names = ("B", "O")
    return (scores / k).sum()


def mean_average_recall_at_k(act: torch.Tensor,
                             pred: torch.Tensor,
                             k: int = 10) -> torch.Tensor:
    r"""Calculate the mean of average recall at k between two tensors of batch of items, 
    i.e. `torch.tensor([[1, 2, 3], [2, 1, 4], ...])` and `torch.tensor([[1, 2, 4, 5], [2, 4, 3, 6], ...])`.
    
    Args:
        act (T), shape = (B, O), dtype = torch.int: A tensor of list of actual items to be predicted
        pred (T), shape = (B, O), dtype = torch.int: A tensor of list of predicted items
        k (int, optional): The maximum number of predicted elements. Defaults to 10.
    
    Returns:
        T: The mean of average recall at k over the batch of items.
    """
    # check if k is smaller than number of outputs's items
    k = pred.size("O") if k > pred.size("O") else k

    # get batch size
    b = pred.size("B")

    # create zeros-like tensor to store values
    scores = torch.zeros(size=(b, 1))
    num_hits = torch.zeros(size=(b, 1))

    # loop through k
    for i in range(k):
        # get pred[:, i] and reshape back to (B, O)
        p = pred[:, i].unflatten("B", [("B", b), ("O", 1)])
        # if pred[row, i] not in pred[row, :i], return True, else False
        not_p = ~(p == pred[:, :i]).sum(dim="O", keepdim=True).bool()
        # if pred[row, i] in act[row, :], return True, else False
        hits = (p == act).sum(dim="O", keepdim=True).bool()
        # sum not_p and hit and convert to integer
        hits = (hits.rename(None) & not_p.rename(None)).int()

        # add hit to score and num_hits
        num_hits += hits
        score = num_hits.masked_fill(~(hits.bool()), 0) / (i + 1)
        scores += score

    ## scores.names = ("B", "O")
    return (scores / act.size("O")).sum()


def discounted_cumulative_gain(y: torch.Tensor,
                               k: int = 10,
                               gain_type: str = "exp2"):
    # Calculate cumulative gain
    y_partial = y[:k]
    if gain_type == "exp2":
        gains = torch.pow(y_partial, 2.0) - 1.0
    elif gain_type == "identity":
        gains = y_partial
    else:
        raise ValueError("gain type only allow \"exp2\" or \"identity\".")

    # Calculate discount
    ranges = torch.arange(1, k + 1, 1).float()
    discount = torch.log2(ranges + 1)
    dcg = torch.sum(torch.div(gains, discount))

    return dcg


def ideal_discounted_cumulative_gain(y: torch.Tensor,
                                     k: int = 10,
                                     gain_type: str = "exp2"):
    # Sort y to an ideal case 
    y_sorted = torch.sort(y, descending=True).values

    # Calculate dcg of y_sorted
    return discounted_cumulative_gain(y_sorted, k, gain_type)


def normalized_discounted_cumulative_gain(y: torch.Tensor,
                                          k: int = 10,
                                          gain_type: str = "exp2"):
    # Calculate dcg of y
    dcg = discounted_cumulative_gain(y, k, gain_type)

    # Calculate idcg of y
    idcg = ideal_discounted_cumulative_gain(y, k, gain_type)

    # Calculate ndcg of y
    return dcg / idcg
