import sys, argparse

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('bst', choices=['xgb', 'lgb', 'cab'])
    parser.add_argument('learning_task', choices=['classification', 'regression'])
    parser.add_argument('-t', '--n_estimators', type=int, default=5000)
    parser.add_argument('-n', '--hyperopt_evals', type=int, default=50)
    parser.add_argument('-s', '--time_sort', type=int, default=None)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--cd', required=True)
    parser.add_argument('-o', '--output_folder_path', default=None)
    parser.add_argument('--holdout_size', type=float, default=0)
    return parser

experiments = [{'name': 'Click',
                'learning_task':'classification',
                'n_estimators':10, 
                'hyperopt_evals': 10, 
                'holdout_size':0,
                'time_sort':None,
                'train':'quality_benchmarks/prepare_click/train',
                'test':'quality_benchmarks/prepare_click/test',
                'cd':'quality_benchmarks/prepare_click/train.cd',
                'output_folder_path':'quality_benchmarks/prepare_click/',
                'bst':'xgb'},
                {'name': 'Amazon',
                'learning_task':'classification',
                'n_estimators':10, 
                'hyperopt_evals': 10, 
                'holdout_size':0,
                'time_sort':None,
                'train':'quality_benchmarks/prepare_amazon/train',
                'test':'quality_benchmarks/prepare_amazon/test',
                'cd':'quality_benchmarks/prepare_amazon/train.cd',
                'output_folder_path':'quality_benchmarks/prepare_amazon/',
                'bst':'xgb'},
                {'name': 'Internet',
                'learning_task':'classification',
                'n_estimators':10, 
                'hyperopt_evals': 10, 
                'holdout_size':0,
                'time_sort':None,
                'train':'quality_benchmarks/prepare_internet/train',
                'test':'quality_benchmarks/prepare_internet/test',
                'cd':'quality_benchmarks/prepare_internet/train.cd',
                'output_folder_path':'quality_benchmarks/prepare_internet/',
                'bst':'xgb'},
                {'name': 'Kick',
                'learning_task':'classification',
                'n_estimators':10, 
                'hyperopt_evals': 10, 
                'holdout_size':0,
                'time_sort':None,
                'train':'quality_benchmarks/prepare_kick/train',
                'test':'quality_benchmarks/prepare_kick/test',
                'cd':'quality_benchmarks/prepare_kick/train.cd',
                'output_folder_path':'quality_benchmarks/prepare_kick/',
                'bst':'xgb'}
                ]


if __name__ == "__main__":
    #parser = createParser()
    #namespace = parser.parse_args(sys.argv[1:])

    for experiment in experiments:
        print("Running experiment: {}".format(experiment['name']))
        if experiment['bst'] == 'xgb':
            from xgboost_experiment import XGBExperiment
            Experiment = XGBExperiment
        elif experiment['bst'] == 'lgb':
            from lightgbm_experiment import LGBExperiment
            Experiment = LGBExperiment
        elif experiment['bst'] == 'cab':
            from catboost_experiment import CABExperiment
            Experiment = CABExperiment

        exp = Experiment(experiment['learning_task'], 
                         experiment['n_estimators'], 
                         experiment['hyperopt_evals'], 
                         experiment['time_sort'], 
                         experiment['holdout_size'], 
                         experiment['train'], 
                         experiment['test'], 
                         experiment['cd'],
                         experiment['output_folder_path'])
        exp.run()
