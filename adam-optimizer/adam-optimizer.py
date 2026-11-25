import numpy as np

def adam_step(
    param,
    grad,
    m,
    v,
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert inputs to float NumPy arrays to preserve shape and precision
    param_arr = np.asarray(param, dtype=np.float64)
    grad_arr = np.asarray(grad, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    # Step 1: Update biased first moment estimate
    m_new = beta1 * m_arr + (1.0 - beta1) * grad_arr

    # Step 2: Update biased second raw moment estimate
    v_new = beta2 * v_arr + (1.0 - beta2) * (grad_arr ** 2)

    # Step 3: Compute bias-corrected moment estimates
    m_hat = m_new / (1.0 - (beta1 ** t))
    v_hat = v_new / (1.0 - (beta2 ** t))

    # Step 4: Update parameters
    param_new = param_arr - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new
