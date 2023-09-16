import json
import logging
import os
import pickle
from typing import Tuple, List, Union, Type, Callable, Dict, Any

import filelock
import gin
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers, Callback
from pytorch_lightning.metrics import Metric
from torch.utils.data import DataLoader, random_split

import experiments.src.training.training_callbacks as custom_callbacks_module
import experiments.src.training.training_loss_fn as custom_loss_fn_module
import experiments.src.training.training_metrics as custom_metrics_module
import experiments.src.wrappers as experiments_wrappers
import src.huggingmolecules.models as huggingmolecules_models
from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.downloading.downloading_utils import HUGGINGMOLECULES_CACHE
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase
from .training_callbacks import NeptuneCompatibleCallback, \
    GinConfigSaver, ModelConfigSaver
from ..gin.gin_utils import get_formatted_config_str, parse_gin_str, get_default_experiment_name

default_cache_dir = os.path.join(HUGGINGMOLECULES_CACHE, 'encodings')
HUGGINGMOLECULES_ENCODINGS_CACHE = os.getenv("HUGGINGMOLECULES_ENCODINGS_CACHE", default_cache_dir)

Split = Dict[str, Dict[str, Union[List, np.array]]]


@gin.configurable('model', denylist=['task'])
class GinModel:
    def __init__(self,
                 cls_name: str,
                 pretrained_name: str,
                 **kwargs):
        self.cls_name = cls_name
        self.pretrained_name = pretrained_name
        self.kwargs = kwargs if kwargs else {}
        self.model_cls = self._get_model_cls()
        self.config_cls = self.model_cls.get_config_cls()
        self.featurizer_cls = self.model_cls.get_featurizer_cls()

    def produce_model(self) -> PretrainedModelBase:
        config = self.produce_config()
        if self.pretrained_name:
            return self.model_cls.from_pretrained(self.pretrained_name, config=config)
        else:
            return self.model_cls(config)

    def produce_featurizer(self) -> PretrainedFeaturizerMixin:
        config = self.produce_config()
        return self.featurizer_cls(config)

    def produce_config(self) -> PretrainedConfigMixin:
        if self.pretrained_name:
            return self.config_cls.from_pretrained(self.pretrained_name, **self.kwargs)
        else:
            return self.config_cls(**self.kwargs)

    def _get_model_cls(self) -> Type[PretrainedModelBase]:
        try:
            model_cls = getattr(huggingmolecules_models, self.cls_name)
        except AttributeError:
            model_cls = getattr(experiments_wrappers, self.cls_name)
            if isinstance(model_cls, ImportError):
                raise model_cls
        return model_cls


def get_default_loggers(save_path: str) -> List[pl_loggers.LightningLoggerBase]:
    return [pl_loggers.CSVLogger(save_path, name='csv_logs')]


def get_default_callbacks() -> List[Callback]:
    gin_config_essential = GinConfigSaver(target_name="gin-config-essential.txt",
                                          excluded_namespaces=['neptune', 'optuna', 'macro', 'benchmark'])
    return [gin_config_essential,
            ModelConfigSaver(),
            GinConfigSaver()]


def get_custom_callbacks(callbacks_names: List[str] = None) -> List[Callback]:
    callbacks_names = callbacks_names if callbacks_names else []
    callbacks = []
    for clb_name in callbacks_names:
        clb_cls = getattr(custom_callbacks_module, clb_name)
        clb = clb_cls()
        callbacks.append(clb)
    return callbacks


@gin.configurable('loss_fn')
def get_loss_fn(*, name, **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    try:
        loss_cls = getattr(custom_loss_fn_module, name)
    except AttributeError:
        loss_cls = getattr(torch.nn, name)
    return loss_cls(**kwargs)


@gin.configurable('metric')
def get_metric_cls(*, name: str, direction: str) -> Type[Metric]:
    try:
        metric_cls = getattr(custom_metrics_module, name)
    except AttributeError:
        metric_cls = getattr(pl.metrics, name)
    setattr(metric_cls, 'direction', direction)
    return metric_cls


@gin.configurable('optimizer', denylist=['model'])
def get_optimizer(model: nn.Module, *, name: str, **kwargs) -> torch.optim.Optimizer:
    opt_cls = getattr(torch.optim, name)
    return opt_cls(model.parameters(), **kwargs)


@gin.configurable('task')
def get_task(name: str) -> str:
    return name


def get_all_hyperparams(model: PretrainedModelBase) -> Dict[str, Any]:
    gin_str = get_formatted_config_str(excluded=['neptune', 'optuna', 'macro', 'benchmark'])
    params = parse_gin_str(gin_str)
    for k, v in model.config.to_dict().items():
        params[f'model.{k}'] = v
    return params


@gin.configurable('neptune', denylist=['model', 'experiment_name'])
def get_neptune_logger(model: PretrainedModelBase, *,
                       user_name: str,
                       project_name: str,
                       experiment_name: str,
                       description: str):
    from pytorch_lightning.loggers import NeptuneLogger
    neptune = NeptuneLogger(api_key=os.environ["NEPTUNE_API_TOKEN"],
                            project_name=f'{user_name}/{project_name}',
                            experiment_name=experiment_name,
                            description=description,
                            tags=[experiment_name],
                            params=get_all_hyperparams(model))
    return neptune


def apply_neptune_logger(neptune_logger, callbacks, loggers):
    loggers += [neptune_logger]
    for clb in callbacks:
        if isinstance(clb, NeptuneCompatibleCallback):
            clb.neptune = neptune_logger


# data

def get_data_loaders(featurizer: PretrainedFeaturizerMixin, *,
                     batch_size: int,
                     num_workers: int = 0,
                     cache_encodings: bool = False,
                     task_name: str = None,
                     dataset_name: str = None,
                     **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if task_name and dataset_name:
        split = get_data_split(task_name=task_name, dataset_name=dataset_name, **kwargs)
    elif task_name is None and dataset_name is None:
        split = get_data_split(**kwargs)
    elif task_name is None:
        split = get_data_split(task_name=task_name, dataset_name=dataset_name, **kwargs)
    else:
        raise AttributeError(
            'Both `task_name` and `dataset_name` attributes must be set either to None or to str values')

    if cache_encodings and _encodings_cached():
        split = _load_encodings_from_cache(split)
    else:
        split['train']['X'] = featurizer.encode_smiles_list(split['train']['X'], split['train']['Y'])
        split['valid']['X'] = featurizer.encode_smiles_list(split['valid']['X'], split['valid']['Y'])
        split['test']['X'] = featurizer.encode_smiles_list(split['test']['X'], split['test']['Y'])

    if cache_encodings and not _encodings_cached():
        _dump_encodings_to_cache(split)

    train_data = split['train']['X']
    valid_data = split['valid']['X']
    test_data = split['test']['X']

    logging.info(f'Train samples: {len(train_data)}')
    logging.info(f'Validation samples: {len(valid_data)}')
    logging.info(f'Test samples: {len(test_data)}')

    train_loader = featurizer.get_data_loader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = featurizer.get_data_loader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = featurizer.get_data_loader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


@gin.configurable('data')
def get_data_split(task_name: str,
                   dataset_name: str,
                   assay_name: str = None,
                   split_method: str = "random",
                   split_frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   split_seed: Union[int, str] = None,
                   normalize_labels: bool = False,
                   dataset_path: str = None) -> dict:
    if dataset_path is None:
        split = _get_data_split_from_tdc(task_name, dataset_name, assay_name,
                                         split_method, split_frac, split_seed)
    else:
        split = _get_data_split_from_csv(dataset_name, assay_name, dataset_path,
                                         split_method, split_frac, split_seed)

    if normalize_labels:
        normalize_labels_inplace(split)

    return split


def _get_data_split_from_tdc(task_name: str,
                             dataset_name: str,
                             assay_name: str,
                             split_method: str,
                             split_frac: Tuple[float, float, float],
                             split_seed: int) -> Split:
    import tdc.single_pred
    task = getattr(tdc.single_pred, task_name)
    data = task(name=dataset_name, label_name=assay_name)
    split = data.get_split(method=split_method, seed=split_seed, frac=split_frac)

    return {
        'train': {'IDs': split['train']['Drug_ID'].to_list(),
                  'X': split['train']['Drug'].to_list(),
                  'Y': split['train']['Y'].to_numpy()},
        'valid': {'IDs': split['valid']['Drug_ID'].to_list(),
                  'X': split['valid']['Drug'].to_list(),
                  'Y': split['valid']['Y'].to_numpy()},
        'test': {'IDs': split['test']['Drug_ID'].to_list(),
                 'X': split['test']['Drug'].to_list(),
                 'Y': split['test']['Y'].to_numpy()},
    }


def _get_data_split_from_csv(dataset_name: str,
                             assay_name: str,
                             dataset_path: str,
                             split_method: str,
                             split_frac: Tuple[float, float, float],
                             split_seed: int) -> Split:
    csv_path = os.path.join(dataset_path, f'{dataset_name.lower()}.csv')
    data = pd.read_csv(csv_path)
    data.insert(0, 'IDs', range(0, len(data)))

    split_path = os.path.join(dataset_path, f'split-{split_method}-{split_seed}.npy')
    if os.path.exists(split_path):
        train_data, valid_data, test_data = _split_data_from_file(data, split_path)
    elif split_method == 'random':
        train_data, valid_data, test_data = _split_data_random(data, split_frac, split_seed)
    else:
        raise NotImplementedError()

    return {
        'train': {'IDs': train_data['IDs'].to_list(),
                  'X': train_data['smiles'].to_list(),
                  'Y': train_data['y'].to_numpy()},
        'valid': {'IDs': valid_data['IDs'].to_list(),
                  'X': valid_data['smiles'].to_list(),
                  'Y': valid_data['y'].to_numpy()},
        'test': {'IDs': test_data['IDs'].to_list(),
                 'X': test_data['smiles'].to_list(),
                 'Y': test_data['y'].to_numpy()}
    }


def _split_data_random(data, split_frac: Tuple[float, float, float], seed: int = None):
    train_len = int(split_frac[0] * len(data))
    valid_len = int(split_frac[1] * len(data))
    test_len = len(data) - train_len - valid_len
    generator = torch.Generator().manual_seed(seed) if seed else None
    train_data, valid_data, test_data = random_split(data, [train_len, valid_len, test_len], generator=generator)
    return data.iloc[train_data.indices], data.iloc[valid_data.indices], data.iloc[test_data.indices]


def _split_data_from_file(data, split_path: str):
    split = np.load(split_path, allow_pickle=True)
    train_split, valid_split, test_split = split.tolist()
    return data.iloc[train_split], data.iloc[valid_split], data.iloc[test_split]


def normalize_labels_inplace(split: Split) -> None:
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler(). \
        fit(np.concatenate([split['train']['Y'], split['valid']['Y']]).reshape(-1, 1))
    split['train']['Y'] = scaler.transform(split['train']['Y'].reshape(-1, 1)).reshape(-1)
    split['valid']['Y'] = scaler.transform(split['valid']['Y'].reshape(-1, 1)).reshape(-1)
    split['test']['Y'] = scaler.transform(split['test']['Y'].reshape(-1, 1)).reshape(-1)


# caching

def _get_encodings_cache_filepath() -> str:
    filename = f'{get_default_experiment_name()}_cache'
    filepath = os.path.join(HUGGINGMOLECULES_ENCODINGS_CACHE, filename)
    return os.path.expanduser(filepath)


def _encodings_cached() -> bool:
    return os.path.exists(_get_encodings_cache_filepath())


def _load_encodings_from_cache(split: Split) -> Split:
    filepath = _get_encodings_cache_filepath()
    lock_path = filepath + '.lock'
    with filelock.FileLock(lock_path):
        with open(filepath, 'rb') as fp:
            data_dict = pickle.load(fp)

    for target in split.keys():
        for i, mol_id in enumerate(split[target]['IDs']):
            split[target]['X'][i] = data_dict[mol_id]

    return split


def _dump_encodings_to_cache(split: Split) -> None:
    data = {}
    for target in split.keys():
        for mol_id, x in zip(split[target]['IDs'], split[target]['X']):
            data[mol_id] = x

    filepath = _get_encodings_cache_filepath()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lock_path = filepath + '.lock'
    with filelock.FileLock(lock_path):
        with open(filepath, 'wb') as fp:
            pickle.dump(data, fp)


# evaluation


def evaluate_and_save_results(trainer: pl.Trainer, test_loader: DataLoader, results_path: str) -> None:
    results = trainer.test(test_dataloaders=test_loader, ckpt_path=None)
    logging.info(results)
    with open(results_path, "w") as f:
        json.dump(results, f)
