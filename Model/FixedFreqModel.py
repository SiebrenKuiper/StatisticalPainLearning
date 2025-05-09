import numpy as np
from scipy.stats import beta

def FixedFreqModel(params, obs_p2 , session, seq):
    """
    认为刺激的频率遵循beta分布  
    p(theta|y_1:t) = Beta(theta| N_1 + 1, N_2 + 1)  
    利用链式法则和马尔可夫性质求出 似然函数 p(y_1:t|theta)  
    p(y_1:t|theta)  
        = p(y_1,y_2,...,y_t|theta)  
        = p(y_1|theta) * p(y_2|y_1,theta) * ... * p(y_t|y_1,...,y_t-1,theta)    
        = p(y_1|theta) * p(y_2|theta) * ... * p(y_t|theta)  
    贝叶斯预测  
    p(y_t+1|y1:t)   
    = ∫ p(y_t+1|y1:t,theta)p(theta|y1:t)dtheta  
    = ∫ p(y_t+1|y_t,theta)p(theta|y1:t)dtheta   
    
    "Window" is the previous n trials where the frequency of stimuli was estimated,     
    "decay" is the previous n trials where the frequency of stimuli further from current trial was discounted following an exponential decay.       """
    window,decay, b1, b2, b3, b4 = params
    for t in range(len(obs_p2)):
        theta = beta.pdf(np.sum(seq[(t-window):t] == 1) + 1,np.sum(seq[(t-window):t] == 2) +1)
