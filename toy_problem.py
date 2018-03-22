import numpy as np
from sklearn.gaussian_process.kernels import Matern
from optml.bayesian_optimizer.gp_categorical import GaussianProcessRegressorWithCategorical
from optml.bayesian_optimizer import BayesianOptimizer
from optml import Parameter
from optml.models import Model
import matplotlib.pyplot as plt

class DummyModel(Model):
    __module__ = 'xgboost'
    def __init__(self):
        pass

def func1(x):
    if x<=1:
        return 2*x
    else:
        return -2*x + 4


def func2(x):
    if x<=1:
        return 2*x + np.sin(x*2*np.pi*10)
    else:
        return -2*x + 4 + np.sin(x*2*np.pi*10)

if __name__ == '__main__':
    # use a toy problem
    optimizer = GaussianProcessRegressorWithCategorical(kernel=Matern(nu=1.5, length_scale_bounds=(0.1, 100.0)),
                                                    alpha=1e-4,
                                                    n_restarts_optimizer=5,
                                                    normalize_y=True)

    func = func2

    xs_truth = np.linspace(0,2,1000)
    ys_truth = [func(x) for x in xs_truth]
    
    bayesOpt = BayesianOptimizer(model=DummyModel(), 
                                 hyperparams=[Parameter(name='bla', param_type='continuous', lower=0.00, upper=2)],
                                 eval_func=log_loss)
    bayesOpt.acquisition_function = 'upper_confidence_bound'
    xs = [[0.05], [0.3]]

    bayesOpt.hyperparam_history.append((xs[0][0], func(xs[0][0])))
    bayesOpt.hyperparam_history.append((xs[1][0], func(xs[1][0])))
    for i in range(15):
        print(i)
        ys = [func(x[0]) for x in xs]
        optimizer.fit(xs,ys)
        minimized = bayesOpt.optimize_continuous_problem(optimizer, [0.1])
        minimized['success']
        minimized['x']
        preds, stds = optimizer.predict(np.array([[x] for x in xs_truth]), return_std=True)
        plt.fill_between(xs_truth, preds-stds, preds+stds, alpha=0.15)
        plt.scatter(xs_truth, preds, alpha=0.3, s=5)

        confs = [bayesOpt.upper_confidence_bound(optimizer, x) for x in xs_truth]
        eis = [bayesOpt.generalized_expected_improvement(optimizer, x, xi=0.1) for x in xs_truth]
        new_x = xs_truth[np.argmax(confs)]
        plt.plot(xs_truth, ys_truth)
        plt.plot(xs_truth, confs)
        plt.plot(xs_truth, eis)
        plt.scatter([x[0] for x in xs], ys)
        plt.show()
        xs.append([new_x])
        bayesOpt.hyperparam_history.append((new_x, func(new_x)))
