from glob import glob
import numpy as np
import pickle as pkl
from sklearn.gaussian_process.kernels import Matern
from optml.bayesian_optimizer.gp_categorical import GaussianProcessRegressorWithCategorical
from optml.bayesian_optimizer import BayesianOptimizer
from optml import Parameter
from optml.models import Model
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

class DummyModel(Model):
    __module__ = 'xgboost'
    def __init__(self):
        pass

params = [
            Parameter(name='eta', param_type='continuous', lower=0.001, upper=1),
            Parameter(name='max_depth', param_type='integer', lower=2, upper=20),
            Parameter(name='subsample', param_type='continuous', lower=0.5, upper=1),
            Parameter(name='colsample_bytree', param_type='continuous', lower=0.5, upper=1),
            Parameter(name='colsample_bylevel', param_type='continuous', lower=0.5, upper=1),
            Parameter(name='min_child_weight', param_type='continuous', lower=0.001, upper=1),
            Parameter(name='alpha', param_type='continuous', lower=0.001, upper=1),
            Parameter(name='lambda', param_type='continuous', lower=0.001, upper=1),
            Parameter(name='gamma', param_type='continuous', lower=0.0, upper=1)
        ]


def eval_func(x,y): 
    return -log_loss(x,y>0.5)

if __name__ == '__main__':
    f = 'quality_benchmarks/prepare_kick/xgb_results_tuned_prepare_kick_20180319-165432.pkl'
    results = pkl.load(open(f, 'r'))
    bayesOpt = BayesianOptimizer(model=DummyModel(), 
                                         hyperparams=params,
                                         eval_func=eval_func)

    xs = np.array([bayesOpt._param_dict_to_arr(x[1]) for x in results['trials']])
    # normalize xs
    #xs -= np.mean(xs,axis=0)
    #stds = np.std(xs,axis=0)
    #xs[:,stds>0] = xs[:,stds>0]/stds[np.newaxis, stds>0]
    ys = [x[0] for x in results['trials']]


    optimizer = GaussianProcessRegressorWithCategorical(kernel=Matern(),
                                                    alpha=1e-4,
                                                    n_restarts_optimizer=5,
                                                    normalize_y=True)

    optimizer.fit(xs,ys)

    for param in params:
        print("{} {}".format(param.name, np.std([bayesOpt._param_arr_to_dict(x)[param.name] for x in xs])))

    param_name = 'colsample_bytree'


    plt.scatter([bayesOpt._param_arr_to_dict(x)[param_name] for x in xs], ys)

    preds, stds = optimizer.predict(xs, return_std=True)
    plt.plot([bayesOpt._param_arr_to_dict(x)[param_name] for x in xs], preds)

    def apply_optimizer(optimizer, X, Y):
        Z = np.zeros(X.shape)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i,j] = optimizer.predict([X[i][j], Y[i][j]])[0]
        return Z

    # plot surface
    for i in range(len(xs)):
        bayesOpt.hyperparam_history.append((ys[i], bayesOpt._param_arr_to_dict(xs[i])))

    print(bayesOpt.get_random_values_arr()); 
    bayesOpt.get_next_hyperparameters(optimizer)
    print(bayesOpt.optimize_continuous_problem(optimizer, bayesOpt.get_random_values_arr())['x'])

    x = np.linspace(params[0].lower, params[0].upper, 30)
    y = np.linspace(params[1].lower, params[1].upper, 30)
    X, Y = np.meshgrid(x, y)
    Z = apply_optimizer(optimizer, X, Y)
    contours = plt.contour(X, Y, Z)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.xlabel(params[0].name)
    plt.ylabel(params[1].name)
    plt.scatter(xs[:,0], xs[:,1], c=range(len(xs)), cmap='Blues')
    plt.show()