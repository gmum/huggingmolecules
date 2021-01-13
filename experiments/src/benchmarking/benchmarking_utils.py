import itertools
import logging
import os
import pickle
from itertools import chain, combinations
from typing import List, Optional, Tuple

import gin
import pandas as pd
import torch

from experiments.src.gin import CONFIGS_ROOT
from experiments.src.gin import get_default_name
from experiments.src.training.training_train_model_utils import get_loss_fn, get_metric_cls, get_data


class EnsembleElement:
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.outputs = {'valid': {}, 'test': {}}

    def __repr__(self):
        return f'{self.name}{self.params}'

    def _cache_repr(self):
        params = '_'.join(f'{k}_{v}' for k, v in sorted(self.params.items()))
        return f'{self.name}_{params}_cached'

    def save(self, cache_dir: str):
        file_path = os.path.join(cache_dir, self._cache_repr())
        with open(file_path, 'wb') as fp:
            pickle.dump(self.outputs, fp)

    def load(self, cache_dir: str):
        logging.info(f'Loading {self} from cache')
        file_path = os.path.join(cache_dir, self._cache_repr())
        with open(file_path, 'rb') as fp:
            self.outputs = pickle.load(fp)

    def is_cached(self, cache_dir: str):
        logging.info(f'Caching {self}')
        file_path = os.path.join(cache_dir, self._cache_repr())
        return os.path.exists(file_path)

    def remove_cache(self, cache_dir: str):
        file_path = os.path.join(cache_dir, self._cache_repr())
        os.remove(file_path)


def get_names_list(prefix_list: Optional[List[str]],
                   models_names_list: Optional[List[str]]):
    models_names_list = models_names_list if models_names_list else [None]
    prefix_list = prefix_list if prefix_list else [None]

    names_list = []
    for prefix, model in itertools.zip_longest(prefix_list, models_names_list):
        if model:
            gin_path = os.path.join(CONFIGS_ROOT, 'models', f'{model}.gin')
            with gin.unlock_config():
                gin.parse_config_file(gin_path)
        prefix = prefix if prefix else gin.query_parameter('name.prefix')
        name = get_default_name(prefix=prefix)
        names_list.append(name)

    return names_list


def get_params_dict():
    params_dict: dict = gin.query_parameter('optuna.params')
    return {k: v.__deepcopy__(None) if isinstance(v, gin.config.ConfigurableReference) else v
            for k, v in params_dict.items()}


def get_params_product_list(params_dict):
    params_dict = {k: list(map(lambda x: (k, x), v)) for k, v in params_dict.items()}
    return [dict(params) for params in itertools.product(*params_dict.values())]


def load_data_from_cache(names_list, cache_dir):
    params_dict = get_params_dict()
    params_dict.pop('data.split_seed')
    hps_list = get_params_product_list(params_dict)
    loaded = []
    missing = []
    for name in names_list:
        for hps in hps_list:
            model = EnsembleElement(name, hps)
            if model.is_cached(cache_dir):
                model.load(cache_dir)
                loaded.append(model)
            else:
                missing.append(model)

    return loaded, missing


def load_targets():
    targets = {'valid': {}, 'test': {}}
    seed_list = get_params_dict()['data.split_seed']
    for seed in seed_list:
        data = get_data(split_seed=seed)
        targets['valid'][seed] = torch.tensor(data['valid']['Y']).float()
        targets['test'][seed] = torch.tensor(data['test']['Y']).float()
    return targets


def fetch_data(names_list: List[str], cache_dir: str) -> Tuple[List[EnsembleElement], dict]:
    os.makedirs(cache_dir, exist_ok=True)
    loaded, missing = load_data_from_cache(names_list, cache_dir)
    if len(missing) > 0:
        fetch_data_from_neptune(missing)
        for model in missing:
            model.save(cache_dir)

    models_list = loaded + missing
    check_models_list(models_list, cache_dir)
    return models_list, load_targets()


def fetch_data_from_neptune(models_list: List[EnsembleElement]) -> None:
    import neptune
    user_name = gin.query_parameter('neptune.user_name')
    project_name = gin.query_parameter('neptune.project_name')
    project = neptune.init(f'{user_name}/{project_name}')

    params_dict = get_params_dict()
    hps_dict = {k: v for k, v in params_dict.items() if k != 'data.split_seed'}

    data = project.get_leaderboard(state='succeeded')
    data = data[data['name'].isin(set(model.name for model in models_list))]
    colums = ['id', 'name'] + [f'parameter_{p}' for p in params_dict]
    data = data[colums]

    for param_name, param_value in params_dict.items():
        dtype = type(param_value[0])
        param_name_pd = f'parameter_{param_name}'
        data[param_name_pd] = data[param_name_pd].astype(dtype) if dtype != int else data[param_name_pd].astype(float).astype(int)

    for model in models_list:
        group = data[data['name'] == model.name]
        for param_name in hps_dict:
            val = model.params[param_name]
            group = group[group[f'parameter_{param_name}'] == val]

        for idx, row in group.iterrows():
            seed = row['parameter_data.split_seed']
            neptune_id = row['id']
            model.outputs['valid'][seed] = get_output_from_artifact(project=project, id=neptune_id,
                                                                    artifact_name='valid_output.pickle')
            model.outputs['test'][seed] = get_output_from_artifact(project=project, id=neptune_id,
                                                                   artifact_name='test_output.pickle')


def get_output_from_artifact(*, project, id: str, artifact_name: str):
    tmp_path = '/tmp/huggingmolecules_experiments/'
    experiment = project.get_experiments(id=id)[0]
    experiment.download_artifact(artifact_name, tmp_path)
    with open(os.path.join(tmp_path, artifact_name), 'rb') as fp:
        output = pickle.load(fp)
    os.system(f'rm -rf {tmp_path}')
    return output


def check_models_list(models_list: List[EnsembleElement], cache_dir: str):
    missing = False
    seed_list = get_params_dict()['data.split_seed']
    for model in models_list:
        for phase in ['valid', 'test']:
            for seed in seed_list:
                if seed not in model.outputs[phase]:
                    missing = True
                    logging.error(f"Model {model} lacks outputs['{phase}'][{seed}].")
                    if model.is_cached(cache_dir):
                        model.remove_cache(cache_dir)
    assert not missing


def pick_best_ensemble(models_list: List[EnsembleElement], *, targets, max_size, names_list, method):
    if method == 'brute':
        return pick_best_ensemble_brute(models_list, targets=targets, max_size=max_size)
    elif method == 'greedy':
        return pick_best_ensemble_greedy(models_list, targets=targets, max_size=max_size)
    elif method == 'all':
        if not max_size:
            return models_list
        size = max_size // len(names_list)
        split = {k: [model for model in models_list if model.name == k][:size] for k in names_list}
        return [model for sublist in split.values() for model in sublist]
    else:
        raise NotImplementedError


def pick_best_ensemble_greedy(models_list: List[EnsembleElement], *, targets, max_size: Optional[int] = None):
    loss_fn = get_loss_fn()
    n = max_size if max_size else len(models_list)

    ensemble = []
    losses = []
    best_loss = float('inf')
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


def pick_best_ensemble_brute(models_list: List[EnsembleElement], *, targets, max_size: Optional[int] = None):
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    loss_fn = get_loss_fn()
    max_size = max_size if max_size else len(models_list)
    ensembles_list = [e for e in powerset(models_list) if 0 < len(e) <= max_size]

    results = []
    for ensemble in ensembles_list:
        valid_loss, _ = evaluate_ensemble(ensemble, targets=targets, phase='valid', metric_fn=loss_fn)
        results.append((valid_loss, ensemble))

    best_valid_loss, best_ensemble = min(results)
    return best_ensemble


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


def print_results(ensemble, targets):
    print(f'Best ensemble:')
    for model in ensemble:
        print(f'  {model}')

    loss_fn = get_loss_fn()
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
