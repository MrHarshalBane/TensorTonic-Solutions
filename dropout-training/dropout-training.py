import numpy as np

def dropout(x, p=0.5, rng=None):
    # Convert input to a NumPy array without modifying the original
    x_arr = np.asarray(x, dtype=float)
    
    # Generate random values in [0.0, 1.0) using the provided rng or fallback
    if rng is not None:
        rand_vals = rng.random(x_arr.shape)
    else:
        rand_vals = np.random.random(x_arr.shape)
        
    # Keep elements where rand_vals >= p (probability 1 - p) and scale by 1 / (1 - p)
    keep_prob = 1.0 - p
    dropout_pattern = (rand_vals >= p).astype(float) / keep_prob
    
    # Apply the scaled mask to the input
    output = x_arr * dropout_pattern
    
    return output, dropout_pattern
