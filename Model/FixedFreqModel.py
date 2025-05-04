import numpy as np

def FixedFreqModel(params, obs_p1, obs_p2, session, seq):
    """
    功能：拟合固定频率模型
    模型描述: 在整个实验中，被试认为刺激强度的频率以一定数值保持恒定的。
    建模思路：
    - 导入待估计参数，包括：
        - p_h: 高疼痛刺激的恒定概率
        - alpha: 概率预测的回归系数
        - beta: session 的回归系数
        - gamma: 残差+截距
        - beta: 截距
        - gamma: session 的影响
    - 计算预测值 y_pred = alpha * p_pred + beta * session + gamma
    - 计算均方误差 mse = np.mean((obs_p2 - y_pred) ** 2) 和 BIC= n * np.log(mse) + k * np.log(n)

    :param obs_p1:低疼痛刺激的频率，由于与高疼痛刺激观测频率线性相关，未在模型中使用。
    :param obs_p2:高疼痛刺激的观测频率。
    :param session:实验中的session，一般有5个seession。
    :param params:
    """
    p_h, alpha, beta, gamma = params
    
    p_pred = np.where(seq == 2, p_h, 1 - p_h)
    
    y_pred = alpha * p_pred + beta * session + gamma
    
    residuals = obs_p2 - y_pred
    mse = np.mean(residuals**2)
    n = len(obs_p2)
    n_params = len(params)
    bic = n * np.log(mse) + n_params * np.log(n)
    
    return bic