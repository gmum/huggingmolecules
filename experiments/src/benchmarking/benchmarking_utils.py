import io
import itertools
import logging
import os
import pickle
import tempfile
from typing import List, Optional, Tuple, Any, Dict

import filelock
import gin
import pandas as pd
import torch

from experiments.src.gin import CONFIGS_ROOT, parse_gin_str
from experiments.src.gin import get_default_name
from experiments.src.training.training_utils import get_data_split
from experiments.src.wrappers.wrappers_molbert import MolbertFeaturizer, MolbertConfig
from src.huggingmolecules.downloading.downloading_utils import HUGGINGMOLECULES_CACHE

default_cache_dir = os.path.join(HUGGINGMOLECULES_CACHE, 'benchmark_results')
HUGGINGMOLECULES_BENCHMARK_CACHE = os.getenv("HUGGINGMOLECULES_BENCHMARK_CACHE", default_cache_dir)


class EnsembleElement:
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.outputs = {'valid': {}, 'test': {}}

    def __repr__(self):
        return f'{self.name}{self.params}'

    @property
    def cache_filename(self):
        params = '_'.join(f'{k}_{v}' for k, v in sorted(self.params.items()))
        return f'{self.name}_{params}_cached'

    @property
    def cache_dir(self):
        return os.path.expanduser(HUGGINGMOLECULES_BENCHMARK_CACHE)

    @property
    def cache_path(self):
        file_path = os.path.join(self.cache_dir, self.cache_filename)
        return os.path.expanduser(file_path)

    @property
    def lock_path(self):
        return f'{self.cache_path}.lock'

    def save_to_cache(self):
        logging.info(f'Caching {self}...')
        os.makedirs(self.cache_dir, exist_ok=True)
        with filelock.FileLock(self.lock_path):
            with open(self.cache_path, 'wb') as fp:
                pickle.dump(self.outputs, fp)

    def load_from_cache(self):
        logging.info(f'Loading {self} from cache...')
        with filelock.FileLock(self.lock_path):
            with open(self.cache_path, 'rb') as fp:
                self.outputs = pickle.load(fp)

    def is_cached(self):
        return os.path.exists(self.cache_path)

    def remove_cache(self):
        with filelock.FileLock(self.lock_path):
            os.remove(self.cache_path)


# fetching data

def fetch_data(names_list: List[str]) -> Tuple[List[EnsembleElement], dict]:
    loaded, missing = _load_data_from_cache(names_list)
    if len(missing) > 0:
        if gin.query_parameter('train.use_neptune'):
            _fetch_data_from_neptune(missing)
        else:
            _fetch_data_locally(missing)
        for model in missing:
            model.save_to_cache()

    models_list = loaded + missing
    check_models_list(models_list)

    if any('MolbertModelWrapper' in item for item in names_list):
        if not all('MolbertModelWrapper' in item for item in names_list):
            raise NotImplementedError('Ensembling molbert with other models is not implemented yet.')
        targets = _load_targets_for_molbert(models_list)
    else:
        targets = _load_targets()
    return models_list, targets


def _load_data_from_cache(names_list):
    params_dict = get_params_dict()
    params_dict.pop('data.split_seed')
    hps_list = _get_params_product_list(params_dict)
    loaded = []
    missing = []
    for name in names_list:
        for hps in hps_list:
            model = EnsembleElement(name, hps)
            if model.is_cached():
                model.load_from_cache()
                loaded.append(model)
            else:
                missing.append(model)

    return loaded, missing


class _UnpicklerCPU(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


# fetching data from neptune

def _fetch_data_from_neptune(models_list: List[EnsembleElement]) -> None:
    import neptune
    user_name = gin.query_parameter('neptune.user_name')
    project_name = gin.query_parameter('neptune.project_name')
    project = neptune.init(f'{user_name}/{project_name}')

    params_dict = get_params_dict()
    hps_dict = {k: v for k, v in params_dict.items() if k != 'data.split_seed'}

    datas = [project.get_leaderboard(state='succeeded', tag=model.name) for model in models_list]
    if all(not data.empty for data in datas):
        data = pd.concat(datas)
    else:
        data = project.get_leaderboard(state='succeeded')
        data = data[data['name'].isin(set(model.name for model in models_list))]

    colums = ['id', 'name'] + [f'parameter_{p}' for p in params_dict]
    data = data[colums]

    for param_name, param_value in params_dict.items():
        dtype = type(param_value[0])
        param_name_pd = f'parameter_{param_name}'
        data[param_name_pd] = data[param_name_pd].astype(dtype) if dtype != int else data[param_name_pd].astype(
            float).astype(int)

    for model in models_list:
        group = data[data['name'] == model.name]
        for param_name in hps_dict:
            val = model.params[param_name]
            group = group[group[f'parameter_{param_name}'] == val]

        for idx, row in group.iterrows():
            seed = row['parameter_data.split_seed']
            neptune_id = row['id']
            model.outputs['valid'][seed] = _fetch_output_from_neptune(project=project, id=neptune_id,
                                                                      artifact_name='valid_output.pickle')
            model.outputs['test'][seed] = _fetch_output_from_neptune(project=project, id=neptune_id,
                                                                     artifact_name='test_output.pickle')


def _fetch_output_from_neptune(*, project, id: str, artifact_name: str) -> Any:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = project.get_experiments(id=id)[0]
            experiment.download_artifact(artifact_name, tmp)
            output = _fetch_output_locally(os.path.join(tmp, artifact_name))
    except Exception as e:
        raise RuntimeError(f'Downloading artifacts failed on {id}. Exception: {e}')
    return output


# fetching data locally

def _fetch_output_locally(output_path: str) -> Any:
    with open(output_path, 'rb') as fp:
        return _UnpicklerCPU(fp).load()


def _fetch_data_locally(models_list: List[EnsembleElement]) -> None:
    data_dict = {}
    for model_name in set(model.name for model in models_list):
        data_dict.update(_fetch_data_dict_locally(model_name))

    for model in models_list:
        seed_list = get_params_dict()['data.split_seed']
        for seed in seed_list:
            params = {k: v for k, v in model.params.items()}
            params.update({'name': model.name, 'data.split_seed': seed})
            outputs = data_dict[frozenset(params.items())]
            model.outputs['valid'][seed] = outputs['valid']
            model.outputs['test'][seed] = outputs['test']


def _fetch_data_dict_locally(model_name: str) -> Dict[frozenset, dict]:
    data_dict = {}
    params_dict = get_params_dict()
    root_path = gin.query_parameter('optuna.root_path')
    save_path = os.path.join(root_path, model_name)
    for path in os.listdir(save_path):
        trial_path = os.path.join(save_path, path)
        if os.path.isdir(trial_path):
            experiment_path = os.path.join(trial_path, model_name)

            gin_path = os.path.join(experiment_path, "gin-config-essential.txt")
            with open(gin_path, 'r') as fp:
                gin_str = fp.read()
            gin_dict = parse_gin_str(gin_str)
            params = {k: v for k, v in gin_dict.items() if k in params_dict.keys()}
            params.update({'name': model_name})

            outputs = {
                'valid': _fetch_output_locally(os.path.join(experiment_path, 'valid_output.pickle')),
                'test': _fetch_output_locally(os.path.join(experiment_path, 'test_output.pickle'))
            }

            data_dict[frozenset(params.items())] = outputs

    return data_dict


# loading targets

def _load_targets() -> Dict[str, Dict[int, torch.FloatTensor]]:
    targets = {'valid': {}, 'test': {}}
    seed_list = get_params_dict()['data.split_seed']
    for seed in seed_list:
        data = get_data_split(split_seed=seed)
        targets['valid'][seed] = torch.tensor(data['valid']['Y']).float()
        targets['test'][seed] = torch.tensor(data['test']['Y']).float()
    return targets


def _load_targets_for_molbert(models_list: List[EnsembleElement]) -> Dict[str, Dict[int, torch.FloatTensor]]:
    targets = {'valid': {}, 'test': {}}
    seed_list = get_params_dict()['data.split_seed']
    featurizer = MolbertFeaturizer(MolbertConfig())
    for seed in seed_list:
        data = get_data_split(split_seed=seed)
        targets['valid'][seed] = featurizer(data['valid']['X'], data['valid']['Y']).y.view(-1)

        batch = featurizer(data['test']['X'], data['test']['Y'])
        test_y = batch.y.view(-1)
        invalid_y = batch.invalid_y
        if invalid_y.shape[0] > 0:
            print(f'Missing {invalid_y.shape[0]} molecules from test set')
            major = torch.mean(test_y)
            test_y = torch.cat([test_y, torch.tensor(invalid_y).float()])
            for model in models_list:
                if 'MolbertModelWrapper' in model.name:
                    output = model.outputs['test'][seed]
                    model.outputs['test'][seed] = torch.cat([output, major.expand_as(invalid_y)])

        targets['test'][seed] = test_y

    return targets


# helpers

def check_models_list(models_list: List[EnsembleElement]) -> None:
    missing = False
    seed_list = get_params_dict()['data.split_seed']
    for model in models_list:
        for phase in ['valid', 'test']:
            for seed in seed_list:
                if seed not in model.outputs[phase]:
                    missing = True
                    logging.error(f"Model {model} lacks outputs['{phase}'][{seed}].")
                    if model.is_cached():
                        model.remove_cache()
    assert not missing


def get_names_list(prefix_list: Optional[List[str]],
                   models_names_list: Optional[List[str]]) -> List[str]:
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


def get_params_dict() -> Dict[str, Any]:
    params_dict: dict = gin.query_parameter('optuna.params')
    return {k: v.__deepcopy__(None) if isinstance(v, gin.config.ConfigurableReference) else v
            for k, v in params_dict.items()}


def _get_params_product_list(params_dict) -> List[dict]:
    params_dict = {k: list(map(lambda x: (k, x), v)) for k, v in params_dict.items()}
    return [dict(params) for params in itertools.product(*params_dict.values())]
