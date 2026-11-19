import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation (provided helper)."""
    z = np.clip(z, -500, 500)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 500
) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    N, D = X.shape
    w = np.zeros(D, dtype=np.float64)
    b = 0.0

    for _ in range(steps):
        # 1. Forward pass
        z = X @ w + b
        p = _sigmoid(z)

        # 2. Compute gradients
        error = p - y
        dw = (X.T @ error) / N
        db = float(np.mean(error))

        # 3. Gradient descent update
        w -= lr * dw
        b -= lr * db

    return w, float(b)
