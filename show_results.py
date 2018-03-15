from glob import glob
import numpy as np
import pickle as pkl

scores_names = ['Catboost Tuned', 'Catboost Default', 'LightGBM Tuned', 'LightGBM Default',
                'XGBoost Tuned', 'XGBoost Default', 'XGBoost Optml']

scores = {
    "Adult": [0.26974,0.27298,0.27602,0.28716,0.27542,0.28009],
    "amazon": [0.13772,0.13811,0.16360,0.16716,0.16327,0.16536],
    "click": [0.39090,0.39112,0.39633,0.39749,0.39624,0.39764],
    "KDD appetency": [0.07151,0.07138,0.07179,0.07482,0.07176,0.07466],
    "KDD churn": [0.23129,0.23193,0.23205,0.23565,0.23312,0.23369],
    "internet": [0.20875,0.22021,0.22315,0.23627,0.22532,0.23468],
    "KDD upselling": [0.16613,0.16674,0.16682,0.17107,0.16632,0.16873],
    "KDD 98": [0.19467,0.19479,0.19576,0.19837,0.19568,0.19795],
    "kick": [0.28479,0.28491,0.29566,0.29877,0.29465,0.29816]
}

def create_result_table(scores):
    col_names = scores_names
    table = '| |' + ' | '.join(col_names) + '|\n'
    for task, score in scores.items():
        score = [round(s,6) for s in score]
        if len(score)<len(col_names):
            score += ['N/A'] * (len(col_names)-len(score))
        table += '|' +task+'|' + '|'.join([str(sc)[:7] for sc in score]) + '|\n'
    return table

if __name__ == '__main__':
    result_files = glob("quality_benchmarks/*/*.pkl")
    experiments = np.unique([res.split('/')[1] for res in result_files])
    model = 'xgb'
    result_dict = {}
    for experiment in experiments:
        experiment_files = [f for f in glob("quality_benchmarks/{}/*.pkl".format(experiment)) if model in f]
        recent_file = sorted(experiment_files, key=lambda x: x[-19:-4])[-1]
        results = pkl.load(open(recent_file, 'r'))
        if 'test' in results.keys():
            mean_loss = np.mean(results['test']['losses'])
            var_loss = np.var(results['test']['losses'])
        else:
            mean_loss = np.mean(results['test_losses'])
            var_loss = np.var(results['test_losses'])
        result_dict[experiment.replace('prepare_','')] = {'loss': mean_loss, 'var': var_loss}

    for k,v in result_dict.items():
        scores[k].append(v['loss'])
    print(create_result_table(scores))