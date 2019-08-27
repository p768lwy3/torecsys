import operator as op
from functools import reduce

def combination(n: int, r: int) -> int:
    r"""[summary]
    
    Args:
        n (int): [description]
        r (int): [description]
    
    Returns:
        int: [description]
    """
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return int(numer / denom)
