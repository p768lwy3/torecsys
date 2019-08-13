from functools import wraps
import warnings

def no_jit_experimental(func: callable):
    pass


def jit_experimental(func: callable):
    """a decorator to write a message in a layer or a estimator where they have 
    been checked to be compatible with torch.jit.trace
    
    Args:
        func (callable): a callable function, and most likely they are torecsys.layers or torecsys.estimators
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("the module have been checked with torch.jit.trace, but the feature is in experimemtal.")
        return func(*args, **kwargs)

    return wrapper
