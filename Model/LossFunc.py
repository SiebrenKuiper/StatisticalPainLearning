from scipy.optimize import minimize

from Model.FixedFreqModel import FixedFreqModel

def fit_model(obs_p1, obs_p2, session, seq, initial_params=None, param_bounds=None):


    if initial_params is None:
        initial_params = [0.5, 0.5, 0.5, 0.5]
    if param_bounds is None:
        param_bounds = [(0, 1), 
                       (None, None), (None, None), (None, None)]

    def bic_objective(params, *args):
        op1, op2, sess, sq = args
        return FixedFreqModel(
            obs_p1=op1,
            obs_p2=op2,
            session=sess,
            seq=sq,
            params=params
        )

    optimization_result = minimize(
        fun=bic_objective,
        x0=initial_params, 
        args=(obs_p1, obs_p2, session, seq),
        method='L-BFGS-B',
        bounds=param_bounds,
        options={'disp': True}
    )
    
    return optimization_result