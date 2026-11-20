import numpy as np

def pad_sequences(seqs: list[list[int]], pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    # Handle empty input edge case
    if not seqs:
        return np.empty((0, 0), dtype=int)
    
    # Determine target sequence length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
        
    # Pre-allocate the output array filled with pad_value
    padded = np.full((len(seqs), max_len), pad_value, dtype=int)
    
    # Copy each sequence into the pre-allocated array (truncating if needed)
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]
        if trunc:
            padded[i, :len(trunc)] = trunc
            
    return padded
