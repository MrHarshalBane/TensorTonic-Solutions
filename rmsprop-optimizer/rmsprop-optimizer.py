import numpy as np

def rmsprop_step(w, g, s, lr=0.1, beta=0.9, eps=1e-8):
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    
    # Step 1: Update running squared gradient average
    new_s = beta * s + (1.0 - beta) * (g ** 2)
    
    # Step 2: Update parameters
    new_w = w - (lr / np.sqrt(new_s + eps)) * g
    
    # Round outputs to 6 decimal places
    return np.round(new_w, 6), np.round(new_s, 6)
