import json
import logging
import os
from typing import Tuple, List, Optional, Union

import gin
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers, Callback
from torch.nn import functional as F
from torch.utils.data import DataLoader

import experiments.training.training_callbacks as custom_callbacks_module
import experiments.training.training_loss_fn as custom_loss_fn_module
import experiments.wrappers as wrappers
import src.huggingmolecules.models as models
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase
from src.huggingmolecules.utils import get_formatted_config_str, parse_gin_str
from .training_callbacks import NeptuneCompatibleCallback, \
    GinConfigSaver, ModelConfigSaver, ConfigurableModelCheckpoint


def get_default_loggers(save_path: str) -> List[pl_loggers.LightningLoggerBase]:
    return [pl_loggers.CSVLogger(save_path)]


@gin.configurable('callbacks', blacklist=['save_path'])
def get_default_callbacks(save_path: str, excluded: List[str] = None) -> List[Callback]:
    checkpoint_callback = ConfigurableModelCheckpoint(filepath=os.path.join(save_path, "weights"))
    gin_config_essential = GinConfigSaver(target_name="gin-config-essential.txt",
                                          excluded_namespaces=['neptune', 'optuna', 'macro'])
    callbacks = [checkpoint_callback, gin_config_essential, ModelConfigSaver(), GinConfigSaver()]
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
def get_loss_fn(*, name='mse_loss', **kwargs):
    try:
        return getattr(custom_loss_fn_module, name)
    except AttributeError:
        return getattr(F, name)


@gin.configurable('optimizer', blacklist=['model'])
def get_optimizer(model: nn.Module, *, name: str = 'Adam', **kwargs) -> torch.optim.Optimizer:
    opt_cls = getattr(torch.optim, name)
    return opt_cls(model.parameters(), **kwargs)


def get_all_hyperparams(model: PretrainedModelBase):
    gin_str = get_formatted_config_str(excluded=['neptune', 'optuna', 'macro'])
    params = parse_gin_str(gin_str)
    for k, v in model.get_config().get_dict().items():
        params[f'model.{k}'] = v
    return params


@gin.configurable('neptune', blacklist=['model', 'save_path'])
def get_neptune(model: PretrainedModelBase,
                save_path: str,
                project_name: str,
                user_name: str,
                experiment_name: Optional[str] = None):
    from pytorch_lightning.loggers import NeptuneLogger
    api_token = os.environ["NEPTUNE_API_TOKEN"]
    description = os.path.basename(save_path)
    params = get_all_hyperparams(model)
    neptune = NeptuneLogger(api_key=api_token,
                            project_name=f'{user_name}/{project_name}',
                            experiment_name=experiment_name if experiment_name else description,
                            description=description,
                            params=params)
    return neptune


def apply_neptune(model: PretrainedModelBase,
                  callbacks: List[Callback],
                  loggers: List[pl_loggers.LightningLoggerBase], *,
                  save_path):
    neptune = get_neptune(model, save_path)
    loggers += [neptune]
    for clb in callbacks:
        if isinstance(clb, NeptuneCompatibleCallback):
            clb.neptune = neptune


@gin.configurable('data', blacklist=['featurizer'])
def get_data_loaders(featurizer: PretrainedFeaturizerMixin, *,
                     batch_size: int,
                     task_name: str,
                     dataset_name: str,
                     split_method: str = "random",
                     split_frac: List[float],
                     split_seed: Union[int, str] = "benchmark",
                     num_workers: int = 0,
                     normalize: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    import tdc.single_pred
    task = getattr(tdc.single_pred, task_name)
    data = task(name=dataset_name)
    split = data.get_split(method=split_method, seed=split_seed, frac=split_frac)

    train_y = split['train']['Y'].to_numpy()
    valid_y = split['valid']['Y'].to_numpy()
    test_y = split['test']['Y'].to_numpy()

    if normalize:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(np.concatenate([train_y, valid_y]).reshape(-1, 1))
        train_y = scaler.transform(train_y.reshape(-1, 1)).reshape(-1)
        valid_y = scaler.transform(valid_y.reshape(-1, 1)).reshape(-1)
        test_y = scaler.transform(test_y.reshape(-1, 1)).reshape(-1)

    train_data = featurizer.encode_smiles_list(split['train']['Drug'], train_y)
    valid_data = featurizer.encode_smiles_list(split['valid']['Drug'], valid_y)
    test_data = featurizer.encode_smiles_list(split['test']['Drug'], test_y)

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


@gin.configurable('model')
def get_model_and_featurizer(cls_name: str, pretrained_name: str):
    try:
        model_cls = getattr(models, cls_name)
    except AttributeError:
        model_cls = getattr(wrappers, cls_name)
    model: PretrainedModelBase = model_cls.from_pretrained(pretrained_name)
    featurizer_cls = model.get_featurizer_cls()
    featurizer: PretrainedFeaturizerMixin = featurizer_cls.from_pretrained(pretrained_name)
    return model, featurizer
