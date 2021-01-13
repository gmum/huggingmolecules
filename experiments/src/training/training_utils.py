import json
import logging
import os
from typing import Tuple, List, Union, Literal

import gin
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers, Callback
from torch.utils.data import DataLoader

import experiments.src.training.training_callbacks as custom_callbacks_module
import experiments.src.training.training_loss_fn as custom_loss_fn_module
import experiments.src.training.training_metrics as custom_metrics_module
import experiments.src.wrappers as wrappers
import src.huggingmolecules.models as models
from experiments.src.gin import get_formatted_config_str, parse_gin_str
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
def get_neptune(model: PretrainedModelBase, *,
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


def apply_neptune(model: PretrainedModelBase,
                  callbacks: List[Callback],
                  loggers: List[pl_loggers.LightningLoggerBase], *,
                  neptune_experiment_name: str,
                  neptune_description: str):
    neptune = get_neptune(model, experiment_name=neptune_experiment_name, description=neptune_description)
    loggers += [neptune]
    for clb in callbacks:
        if isinstance(clb, NeptuneCompatibleCallback):
            clb.neptune = neptune


@gin.configurable('data')
def get_data(task_name: str,
             dataset_name: str,
             split_method: str = "random",
             split_frac: List[float] = (0.8, 0.1, 0.1),
             split_seed: Union[int, str] = "benchmark",
             normalize_labels: bool = False):
    import tdc.single_pred
    task = getattr(tdc.single_pred, task_name)
    data = task(name=dataset_name)
    split = data.get_split(method=split_method, seed=split_seed, frac=split_frac)

    train_y = split['train']['Y'].to_numpy()
    valid_y = split['valid']['Y'].to_numpy()
    test_y = split['test']['Y'].to_numpy()

    if normalize_labels:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(np.concatenate([train_y, valid_y]).reshape(-1, 1))
        train_y = scaler.transform(train_y.reshape(-1, 1)).reshape(-1)
        valid_y = scaler.transform(valid_y.reshape(-1, 1)).reshape(-1)
        test_y = scaler.transform(test_y.reshape(-1, 1)).reshape(-1)

    return {
        'train': {'X': split['train']['Drug'], 'Y': train_y},
        'valid': {'X': split['valid']['Drug'], 'Y': valid_y},
        'test': {'X': split['test']['Drug'], 'Y': test_y},
    }


def get_data_loaders(featurizer: PretrainedFeaturizerMixin, *,
                     batch_size: int,
                     num_workers: int = 0, ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data = get_data()

    train_data = featurizer.encode_smiles_list(data['train']['X'], data['train']['Y'])
    valid_data = featurizer.encode_smiles_list(data['valid']['X'], data['valid']['Y'])
    test_data = featurizer.encode_smiles_list(data['test']['X'], data['test']['Y'])

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

    def get_model_cls(self):
        try:
            model_cls = getattr(models, self.cls_name)
        except AttributeError:
            model_cls = getattr(wrappers, self.cls_name)
        return model_cls

    def get_model(self):
        model_cls = self.get_model_cls()
        return model_cls.from_pretrained(self.pretrained_name, self.task, **self.kwargs)

    def get_featurizer(self):
        model_cls = self.get_model_cls()
        featurizer_cls = model_cls.get_featurizer_cls()
        return featurizer_cls.from_pretrained(self.pretrained_name)
