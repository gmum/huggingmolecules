import itertools
import json
import logging
import os
import tempfile
from typing import List, Any, Dict

import gin
import numpy as np

from experiments.src.gin import get_default_name, parse_gin_str
from experiments.src.training.training_utils import get_metric_cls


def get_grid_results_dict():
    results_dict = create_results_dict()
    study_name = get_default_name()
    if gin.query_parameter('train.use_neptune'):
        fetch_results_dict_from_neptune(results_dict, study_name)
    else:
        fetch_results_dict_from_local(results_dict, study_name)
    return results_dict


def create_results_dict():
    seeds = get_params_dict()['data.split_seed']
    hps_dict = get_hps_dict()
    hps_product_list = _get_params_product_list(hps_dict)
    return {frozenset(hps.items()): {seed: None} for seed in seeds for hps in hps_product_list}


# neptune

def fetch_results_dict_from_neptune(results_dict, study_name):
    import neptune
    user_name = gin.query_parameter('neptune.user_name')
    project_name = gin.query_parameter('neptune.project_name')
    project = neptune.init(f'{user_name}/{project_name}')

    df = download_dataframe_from_neptune(project, study_name)

    hps_names = list(get_hps_dict().keys())
    for idx, row in df.iterrows():
        hps = dict(row[hps_names])
        seed = int(row['data.split_seed'])
        results = download_results_from_neptune(project, row['id'], 'results.json')
        results_dict[frozenset(hps.items())][seed] = results


def download_dataframe_from_neptune(neptune_project, name: str):
    df = neptune_project.get_leaderboard(state='succeeded', tag=name)

    params_dict = get_params_dict()
    df.rename(columns={f'parameter_{p}': p for p in params_dict.keys()}, inplace=True)
    df = df[['id', 'name'] + list(params_dict.keys())]

    for param_name, param_value in params_dict.items():
        dtype = type(param_value[0])
        df[param_name] = df[param_name].astype(dtype) if dtype != int else df[param_name].astype(float).astype(int)

    return df


def download_results_from_neptune(project, id: str, artifact_name: str) -> Any:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = project.get_experiments(id=id)[0]
            experiment.download_artifact(artifact_name, tmp)
            output = load_results_from_local(os.path.join(tmp, artifact_name))
    except Exception as e:
        raise RuntimeError(f'Downloading artifacts failed on {id}. Exception: {e}')
    return output


# local

def fetch_results_dict_from_local(results_dict: Dict[frozenset, Dict[int, dict]], study_name: str):
    root_path = gin.query_parameter('optuna.root_path')
    save_path = os.path.join(root_path, study_name)

    hps_names = set(get_hps_dict().keys())
    for path in os.listdir(save_path):
        trial_path = os.path.join(save_path, path)
        if os.path.isdir(trial_path):
            experiment_path = os.path.join(trial_path, study_name)
            gin_path = os.path.join(experiment_path, "gin-config-essential.txt")
            with open(gin_path, 'r') as fp:
                gin_str = fp.read()
            gin_dict = parse_gin_str(gin_str)

            hps = {k: v for k, v in gin_dict.items() if k in hps_names}
            seed = gin_dict['data.split_seed']
            results = load_results_from_local(os.path.join(experiment_path, 'results.json'))
            results_dict[frozenset(hps.items())][seed] = results


def load_results_from_local(output_path: str) -> Any:
    return json.load(open(output_path, 'r'))[0]


# results check

def check_grid_results_dict(results_dict: Dict[frozenset, Dict[int, dict]]):
    any_missing_result = False
    for params, results in results_dict.items():
        for seed, result in results.items():
            if result is None:
                any_missing_result = True
                logging.error(f'Results for seed: {seed} and hps: {dict(params)} are missing')
    assert not any_missing_result


# helper methods

def get_params_dict() -> Dict[str, Any]:
    params_dict: dict = gin.query_parameter('optuna.params')
    return {k: v.__deepcopy__(None) if isinstance(v, gin.config.ConfigurableReference) else v
            for k, v in params_dict.items()}


def get_hps_dict() -> Dict[str, Any]:
    params_dict = get_params_dict()
    return {k: v for k, v in params_dict.items() if k != 'data.split_seed'}


def _get_params_product_list(params_dict) -> List[dict]:
    params_dict = {k: list(map(lambda x: (k, x), v)) for k, v in params_dict.items()}
    return [dict(params) for params in itertools.product(*params_dict.values())]


# compute results

def compute_result(results_dict):
    metric_cls = get_metric_cls()
    metric_name = metric_cls.__name__.lower()
    agg_fn = min if metric_cls.direction == 'minimize' else max
    valid_metric = f'valid_{metric_name}'
    test_metric = f'test_{metric_name}'
    results = [(params, average_dictionary(results)) for params, results in results_dict.items()]
    params, result = agg_fn(results, key=lambda x: x[1][valid_metric])
    return params, result, test_metric


def average_dictionary(d: Dict[int, Dict[str, float]]):
    values = list(d.values())[0].keys()
    agg = {v: [d[seed][v] for seed in d.keys()] for v in values}
    mean = {k: np.mean(lst) for k, lst in agg.items()}
    std = {f'{k}_std': np.std(lst) for k, lst in agg.items()}
    return {**mean, **std}


def print_results(params, result, test_metric):
    print(f'Best params: {dict(params)}')
    for key in sorted(key for key in result.keys() if not key.endswith('_std')):
        print(f'\t{key}: {rounded_mean_std(result, key)}')
    print(f'Result: {rounded_mean_std(result, test_metric)}')


def rounded_mean_std(result, key):
    return f'{round(result[key], 3)} \u00B1 {round(result[f"{key}_std"], 3)}'
