import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    # Column vector of positions: shape (seq_len, 1)
    positions = np.arange(seq_len, dtype=float)[:, np.newaxis]
    
    # Even indices [0, 2, 4, ...] for frequency calculation: shape (ceil(d_model / 2),)
    even_indices = np.arange(0, d_model, 2, dtype=float)
    
    # Compute 1 / (base ** (2i / d_model)) in log-space for numerical stability
    div_term = np.exp(-even_indices * (np.log(base) / d_model))
    
    # Outer product via broadcasting: shape (seq_len, ceil(d_model / 2))
    angles = positions * div_term
    
    # Initialize output matrix with float dtype
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # Assign sine to even columns and cosine to odd columns
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
    
    return pe
