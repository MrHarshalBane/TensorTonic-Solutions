import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x_arr = np.asarray(x, dtype=float)
    result = 1.0 / (1.0 + np.exp(-x_arr))
    
    if np.isscalar(x):
        return float(result)
    return result
