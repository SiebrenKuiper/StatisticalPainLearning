from scipy.optimize import differential_evolution

from Model.RandomFreqModel import RandomFreqModel
from Model.RWModel import RWModel



def fit_model(modelname,obs_p2, session, seq):
    result = None
    if modelname == 'RandomFreqModel':
        result = differential_evolution(
            lambda x: RandomFreqModel(x, obs_p2, session, seq),
            bounds=[(0,1)] + [(-10,10)]*3, 
            popsize=15,
            mutation=0.8,
            recombination=0.7,
            disp=True,
            seed = 20243096,
            )
    if modelname == 'RWModel':
        result = differential_evolution(
            lambda x: RWModel(x, obs_p2, session, seq),
            bounds=[(-10,10)]*4, 
            popsize=15,
            mutation=0.8,
            recombination=0.7,
            disp=True,
            seed = 20243096,
            )
    return result