import numpy as np

def RWModel(params, obs_p2 , session, seq):
    alpha, b1, b2, b3 = params
    p = 0.5
    p_pred = []
    for t in range(len(obs_p2 )):
        if seq[t] == 1:
            r = 0
        else:
            r = 1
        p = p + alpha * (r - p)
        p_pred.append(p)
    p_pred = np.array(p_pred)
    y_pred = b1 * p_pred + b2 * session + b3
    residuals = obs_p2 - y_pred
    mse = np.mean(residuals**2)
    n = len(obs_p2)
    n_params = len(params)
    bic = n * np.log(mse) + n_params * np.log(n)
        
    return bic