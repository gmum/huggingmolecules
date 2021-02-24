import json
import logging
import os
import pickle
from typing import Tuple, List, Union, Literal, Type

import gin
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers, Callback
from torch.utils.data import DataLoader, random_split

import experiments.src.training.training_callbacks as custom_callbacks_module
import experiments.src.training.training_loss_fn as custom_loss_fn_module
import experiments.src.training.training_metrics as custom_metrics_module
import experiments.src.wrappers as wrappers
import src.huggingmolecules.models as models
from experiments.src.gin import get_formatted_config_str, parse_gin_str, get_default_name
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase
from .training_callbacks import NeptuneCompatibleCallback, \
    GinConfigSaver, ModelConfigSaver, ConfigurableModelCheckpoint, ModelOutputSaver


def get_default_loggers(save_path: str) -> List[pl_loggers.LightningLoggerBase]:
    return [pl_loggers.CSVLogger(save_path)]


@gin.configurable('callbacks', blacklist=['save_path'])
def get_default_callbacks(save_path: str, excluded: List[str] = None) -> List[Callback]:
    checkpoint_callback = ConfigurableModelCheckpoint(filepath=os.path.join(save_path, "weights"))
    gin_config_essential = GinConfigSaver(target_name="gin-config-essential.txt",
                                          excluded_namespaces=['neptune', 'optuna', 'macro', 'benchmark'])
    callbacks = [checkpoint_callback,
                 gin_config_essential,
                 ModelConfigSaver(),
                 GinConfigSaver(),
                 ModelOutputSaver()]
    excluded = excluded if excluded else []
    callbacks = [clb for clb in callbacks if type(clb).__name__ not in excluded]
    return callbacks


def get_custom_callbacks(callbacks_names: List[str] = None) -> List[Callback]:
    callbacks_names = callbacks_names if callbacks_names else []
    callbacks = []
    for clb_name in callbacks_names:
        clb_cls = getattr(custom_callbacks_module, clb_name)
        clb = clb_cls()
        callbacks.append(clb)
    return callbacks


@gin.configurable('loss_fn')
def get_loss_fn(*, name, **kwargs):
    try:
        loss_cls = getattr(custom_loss_fn_module, name)
    except AttributeError:
        loss_cls = getattr(torch.nn, name)
    return loss_cls(**kwargs)


@gin.configurable('metric')
def get_metric_cls(*, name: str):
    try:
        metric_cls = getattr(custom_metrics_module, name)
    except AttributeError:
        metric_cls = getattr(pl.metrics, name)
    return metric_cls


@gin.configurable('optimizer', blacklist=['model'])
def get_optimizer(model: nn.Module, *, name: str, **kwargs) -> torch.optim.Optimizer:
    opt_cls = getattr(torch.optim, name)
    return opt_cls(model.parameters(), **kwargs)


@gin.configurable('task')
def get_task(name: str):
    return name


def get_all_hyperparams(model: PretrainedModelBase):
    gin_str = get_formatted_config_str(excluded=['neptune', 'optuna', 'macro', 'benchmark'])
    params = parse_gin_str(gin_str)
    for k, v in model.get_config().get_dict().items():
        params[f'model.{k}'] = v
    return params


@gin.configurable('neptune', blacklist=['model', 'experiment_name'])
def _get_neptune(model: PretrainedModelBase, *,
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


def _apply_neptune(model: PretrainedModelBase,
                   callbacks: List[Callback],
                   loggers: List[pl_loggers.LightningLoggerBase], *,
                   neptune_experiment_name: str,
                   neptune_description: str):
    neptune = _get_neptune(model, experiment_name=neptune_experiment_name, description=neptune_description)
    loggers += [neptune]
    for clb in callbacks:
        if isinstance(clb, NeptuneCompatibleCallback):
            clb.neptune = neptune


def _get_data_split_tdc(task_name: str,
                        dataset_name: str,
                        assay_name: str,
                        split_method: str,
                        split_frac: Tuple[float, float, float],
                        split_seed: Union[int, str]):
    import tdc.single_pred
    task = getattr(tdc.single_pred, task_name)
    data = task(name=dataset_name, label_name=assay_name)
    split = data.get_split(method=split_method, seed=split_seed, frac=split_frac)

    return {
        'train': {'IDs': split['train']['Drug_ID'].to_list(), 'X': split['train']['Drug'].to_list(),
                  'Y': split['train']['Y'].to_numpy()},
        'valid': {'IDs': split['valid']['Drug_ID'].to_list(), 'X': split['valid']['Drug'].to_list(),
                  'Y': split['valid']['Y'].to_numpy()},
        'test': {'IDs': split['test']['Drug_ID'].to_list(), 'X': split['test']['Drug'].to_list(),
                 'Y': split['test']['Y'].to_numpy()},
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


def _get_data_split_csv(dataset_name: str,
                        assay_name: str,
                        dataset_path: str,
                        split_method: str,
                        split_frac: Tuple[float, float, float],
                        split_seed: Union[int, str]):
    split_seed = 1234 if split_seed == "benchmark" else split_seed
    csv_path = os.path.join(dataset_path, f'{dataset_name.lower()}.csv')
    data = pd.read_csv(csv_path)
    data.insert(0, 'IDs', range(0, len(data)))
    if split_method == 'random':
        train_data, valid_data, test_data = _split_data_random(data, split_frac, split_seed)
    else:
        if split_seed <= 5:
            split_path = os.path.join(dataset_path, f'split-random-{split_seed}.npy')
            train_data, valid_data, test_data = _split_data_from_file(data, split_path)
        else:
            train_data, valid_data, test_data = _split_data_random(data, split_frac, split_seed)

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


def normalize_labels_inplace(split: dict):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler(). \
        fit(np.concatenate([split['train']['Y'], split['valid']['Y']]).reshape(-1, 1))
    split['train']['Y'] = scaler.transform(split['train']['Y'].reshape(-1, 1)).reshape(-1)
    split['valid']['Y'] = scaler.transform(split['valid']['Y'].reshape(-1, 1)).reshape(-1)
    split['test']['Y'] = scaler.transform(split['test']['Y'].reshape(-1, 1)).reshape(-1)


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
        split = _get_data_split_tdc(task_name, dataset_name, assay_name,
                                    split_method, split_frac, split_seed)
    else:
        split = _get_data_split_csv(dataset_name, assay_name, dataset_path,
                                    split_method, split_frac, split_seed)

    if normalize_labels:
        normalize_labels_inplace(split)

    return split


def _get_cache_filepath():
    filename = f'{get_default_name()}_cache'
    return os.path.join('cached', filename)


def _is_cached():
    return os.path.exists(_get_cache_filepath())


def _load_from_cache(split):
    filepath = _get_cache_filepath()
    with open(filepath, 'rb') as fp:
        data_dict = pickle.load(fp)

    for target in split.keys():
        for i, mol_id in enumerate(split[target]['IDs']):
            split[target]['X'][i] = data_dict[mol_id]

    return split


def _dump_to_cache(split):
    data = {}
    for target in split.keys():
        for mol_id, x in zip(split[target]['IDs'], split[target]['X']):
            data[mol_id] = x

    filepath = _get_cache_filepath()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)


def get_data_loaders(featurizer: PretrainedFeaturizerMixin, *,
                     batch_size: int,
                     num_workers: int = 0,
                     cache: bool = False,
                     task_name: str = None,
                     dataset_name: str = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if task_name and dataset_name:
        split = get_data_split(task_name=task_name, dataset_name=dataset_name)
    elif task_name is None and dataset_name is None:
        split = get_data_split()
    else:
        raise AttributeError(
            'Both `task_name` and `dataset_name` attributes must be set either to None or to str values')

    if cache and _is_cached():
        split = _load_from_cache(split)
    else:
        split['train']['X'] = featurizer.encode_smiles_list(split['train']['X'], split['train']['Y'])
        split['valid']['X'] = featurizer.encode_smiles_list(split['valid']['X'], split['valid']['Y'])
        split['test']['X'] = featurizer.encode_smiles_list(split['test']['X'], split['test']['Y'])

    if cache and not _is_cached():
        _dump_to_cache(split)

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


def evaluate_and_save_results(trainer: pl.Trainer, test_loader: DataLoader, save_path: str, evaluation: str):
    logging.info(f'Running test evaluation for {evaluation} weights')
    ckpt_path = 'best' if evaluation == 'best' else None
    results = trainer.test(test_dataloaders=test_loader, ckpt_path=ckpt_path)
    logging.info(results)
    with open(os.path.join(save_path, "test_results.json"), "w") as f:
        json.dump(results, f)


@gin.configurable('model', blacklist=['task'])
class GinModel:
    def __init__(self,
                 cls_name: str,
                 pretrained_name: str,
                 task: Literal["regression", "classification"],
                 **kwargs):
        self.cls_name = cls_name
        self.pretrained_name = pretrained_name
        self.task = task
        self.kwargs = kwargs if kwargs else {}

    def get_model_cls(self) -> Type[PretrainedModelBase]:
        try:
            model_cls = getattr(models, self.cls_name)
        except AttributeError:
            model_cls = getattr(wrappers, self.cls_name)
        return model_cls

    def get_model(self):
        model_cls = self.get_model_cls()
        if self.pretrained_name is not None:
            return model_cls.from_pretrained(self.pretrained_name, self.task, **self.kwargs)
        else:
            config_cls = model_cls.get_config_cls()
            config = config_cls()
            return model_cls(config)

    def get_featurizer(self) -> PretrainedFeaturizerMixin:
        model_cls = self.get_model_cls()
        featurizer_cls = model_cls.get_featurizer_cls()
        if self.pretrained_name is not None:
            return featurizer_cls.from_pretrained(self.pretrained_name)
        else:
            config_cls = model_cls.get_config_cls()
            config = config_cls()
            return featurizer_cls(config)
