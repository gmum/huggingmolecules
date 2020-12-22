import functools
import os
import pickle
from itertools import chain, combinations
from typing import Optional, List, Literal

import gin
import pandas as pd
import torch

from experiments.src.gin import get_default_name
from experiments.src.training.training_train_model_utils import get_loss_fn, get_data, get_metric_cls


class EnsembleElement:
    def __init__(self, name):
        self.name = name
        self.outputs = {'valid': {}, 'test': {}}

    def __repr__(self):
        return f'EnsembleElement{self.name}'


def fetch_data_from_neptune():
    import neptune
    user_name = gin.query_parameter('neptune.user_name')
    project_name = gin.query_parameter('neptune.project_name')
    experiment_name = get_default_name()
    hps_dict = gin.query_parameter('optuna.params')
    hps_list = [hps for hps in hps_dict.keys() if hps != 'data.split_seed']
    project = neptune.init(f'{user_name}/{project_name}')

    data = project.get_leaderboard(state='succeeded')
    data = data[data['name'] == experiment_name]
    no_trials = len(data)
    total_no_trials = functools.reduce(lambda a, b: a * b,
                                       map(len, (v.__deepcopy__(None)
                                                 if isinstance(v, gin.config.ConfigurableReference)
                                                 else v for v in hps_dict.values())))

    print(f'BENCHMARK PROGRESS: {no_trials}/{total_no_trials}')

    return project, data, hps_list


def get_output_from_artifact(*, project, id: str, artifact_name: str):
    tmp_path = 'tmp'
    experiment = project.get_experiments(id=id)[0]
    experiment.download_artifact(artifact_name, tmp_path)
    with open(os.path.join(tmp_path, artifact_name), 'rb') as fp:
        output = pickle.load(fp)
    os.system(f'rm -rf {tmp_path}')
    return output


def evaluate_ensemble(ensemble: List[EnsembleElement], *, targets: dict, phase: str, metric_fn: callable):
    results = []
    for seed in targets[phase].keys():
        mean_output = torch.mean(torch.stack([e.outputs[phase][seed] for e in ensemble]), dim=0)
        res = metric_fn(mean_output, targets[phase][seed])
        results.append(res)

    results = torch.stack(results)
    mean = float(torch.mean(results))
    std = float(torch.std(results))
    return mean, std


def check_models_list(models_list: List[EnsembleElement], targets):
    error = False
    for model in models_list:
        for phase in ['valid', 'test']:
            for seed in targets[phase].keys():
                try:
                    model.outputs[phase][seed]
                except KeyError:
                    error = True
                    print(f"Model {model} lacks outputs['{phase}'][{seed}].")
    assert not error


def pick_best_ensemble_greedy(models_list: List[EnsembleElement], *, targets, loss_fn,
                              ensemble_max_size: Optional[int] = None):
    ensemble = []
    losses = []
    best_loss = float('inf')
    n = ensemble_max_size if ensemble_max_size else len(models_list)
    for i in range(n):
        for idx, model in enumerate(models_list):
            ensemble.append(model)
            valid_loss, std = evaluate_ensemble(ensemble, targets=targets, phase='valid', metric_fn=loss_fn)
            losses.append((valid_loss, idx))
            ensemble.pop()
        curr_best_loss, curr_best_idx = min(losses)
        if best_loss > curr_best_loss:
            ensemble.append(models_list[curr_best_idx])
            models_list.pop(curr_best_idx)
            best_loss = curr_best_loss
        else:
            break

    return ensemble


def pick_best_ensemble_brute(models_list: List[EnsembleElement], *, targets, loss_fn,
                             ensemble_max_size: Optional[int] = None):
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    ensemble_max_size = ensemble_max_size if ensemble_max_size else len(models_list)
    ensembles_list = [e for e in powerset(models_list) if 0 < len(e) <= ensemble_max_size]
    results = []
    for ensemble in ensembles_list:
        valid_loss, _ = evaluate_ensemble(ensemble, targets=targets, phase='valid', metric_fn=loss_fn)
        results.append((valid_loss, ensemble))

    best_valid_loss, best_ensemble = min(results)
    return best_ensemble


def print_benchmark_results_ensemble(ensemble_max_size: Optional[int] = None,
                                     ensemble_pick_method: Literal['brute', 'greedy'] = 'brute'):
    neptune_project, data, hps_list = fetch_data_from_neptune()

    data['parameter_data.split_seed'] = data['parameter_data.split_seed'].astype(float).astype(int)
    grouped = data.groupby([f'parameter_{hps}' for hps in hps_list] + ['name'])

    models_list = []
    for name, group in grouped:
        model = EnsembleElement(name)
        for idx, row in group[['id', 'parameter_data.split_seed']].iterrows():
            neptune_id = row['id']
            seed = row['parameter_data.split_seed']
            model.outputs['valid'][seed] = get_output_from_artifact(project=neptune_project, id=neptune_id,
                                                                    artifact_name='valid_output.pickle')
            model.outputs['test'][seed] = get_output_from_artifact(project=neptune_project, id=neptune_id,
                                                                   artifact_name='test_output.pickle')
        models_list.append(model)

    targets = {'valid': {}, 'test': {}}
    for seed in data['parameter_data.split_seed'].unique():
        data = get_data(split_seed=seed)
        targets['valid'][seed] = torch.tensor(data['valid']['Y']).float()
        targets['test'][seed] = torch.tensor(data['test']['Y']).float()

    # with open('rest', 'wb') as fp:
    #     pickle.dump((data, models_list, targets), fp)
    #
    # with open('rest', 'rb') as fp:
    #     data, models_list, targets = pickle.load(fp)

    check_models_list(models_list, targets)

    loss_fn = get_loss_fn()
    if ensemble_pick_method == 'brute':
        ensemble = pick_best_ensemble_brute(models_list, targets=targets, loss_fn=loss_fn,
                                            ensemble_max_size=ensemble_max_size)
    elif ensemble_pick_method == 'greedy':
        ensemble = pick_best_ensemble_greedy(models_list, targets=targets, loss_fn=loss_fn,
                                             ensemble_max_size=ensemble_max_size)
    elif ensemble_pick_method == 'all':
        ensemble = models_list if not ensemble_max_size else models_list[:ensemble_max_size]
    else:
        raise NotImplementedError

    print(f'Best ensemble:')
    for model in ensemble:
        print(f'  {model}')

    metric_cls = get_metric_cls()
    metric_name = metric_cls.__name__.lower()

    results = []
    for name, metric, phase in [('valid_loss', loss_fn, 'valid'),
                                ('test_loss', loss_fn, 'test'),
                                (f'valid_{metric_name}', metric_cls(), 'valid'),
                                (f'test_{metric_name}', metric_cls(), 'test')]:
        mean, std = evaluate_ensemble(ensemble, targets=targets, phase=phase, metric_fn=metric)
        results.append((name, mean, std))

    results = pd.DataFrame(results, columns=['name', 'mean', 'std'])
    print(results.groupby(['name']).agg({'mean': 'max', 'std': 'max'}))
    print()
    print(f'Result: {round(mean, 3):.3f} \u00B1 {round(std, 3):.3f}')


def print_benchmark_results_standard(test_metric: str):
    project, data, hps_list = fetch_data_from_neptune()

    valid_metric = f'channel_valid_loss'
    test_metric = f'channel_{test_metric}'

    data[valid_metric] = data[valid_metric].astype(float)
    data[test_metric] = data[test_metric].astype(float)

    grouped = data.groupby([f'parameter_{hps}' for hps in hps_list])
    aggregated = grouped[[valid_metric, test_metric]].agg(['mean', 'std'])
    best_hps = aggregated[valid_metric]['mean'].idxmin()
    result = aggregated.loc[best_hps]

    print(result)
    print('Best hps:')
    best_hps = [best_hps] if not isinstance(best_hps, tuple) else best_hps
    for hps, val in zip(hps_list, best_hps):
        print(f'  {hps} = {val}')
