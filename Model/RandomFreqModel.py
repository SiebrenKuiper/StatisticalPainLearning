import numpy as np

def RandomFreqModel(params, obs_p2, session, seq):
    p_h, b1, b2, b3 = params
    p_pred = np.where(seq == 2, p_h, 1 - p_h)
    y_pred = b1 * p_pred + b2 * session + b3
    residuals = obs_p2 - y_pred
    mse = np.mean(residuals ** 2)
    n = len(obs_p2)
    n_params = len(params)
    bic = n * np.log(mse) + n_params * np.log(n)
    
    return bic