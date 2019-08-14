r"""torecsys.utils.logging.decorator is a sub module of python-decorator functions to tag / label features of functions
"""

from functools import wraps
import warnings


def to_be_tested(func: callable):
    r"""a decorator to write a message in a layer or a estimator where they have not been tested
    
    Args:
        func (callable): a callable function, and most likely they are torecsys.layers or torecsys.estimators
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("this module is in developemnt, and will be tested in the near futures.")
        return func(*args, **kwargs)
    return wrapper


def no_jit_experimental(func: callable):
    r"""a decorator to write a message in a layer or a estimator where they have been checked 
    to be non-compatible with torch.jit.trace
    
    Args:
        func (callable): a callable function, and most likely they are torecsys.layers or torecsys.estimators
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("The module is checked that it is not compatible with torch.jit.trace.")
        return func(*args, **kwargs)
    return wrapper


def jit_experimental(func: callable):
    r"""a decorator to write a message in a layer or a estimator where they have been checked 
    to be compatible with torch.jit.trace
    
    Args:
        func (callable): a callable function, and most likely they are torecsys.layers or torecsys.estimators
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("the module have been checked with torch.jit.trace, but the feature is in experimemtal.")
        return func(*args, **kwargs)
    return wrapper
