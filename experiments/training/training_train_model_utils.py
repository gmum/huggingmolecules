import json
import logging
import os
from typing import Tuple, List, Optional, Union

import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers, Callback
from torch.nn import functional as F
from torch.utils.data import DataLoader

import experiments.training.training_callbacks as custom_callbacks_module
import experiments.training.training_loss_fn as custom_loss_fn_module
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from .training_callbacks import NeptuneCompatibleCallback, HyperparamsNeptuneSaver, \
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


# @gin.configurable('lr_scheduler', blacklist=['optimizer'])
# def get_lr_scheduler(optimizer, name='IdentityScheduler', **kwargs):
#     try:
#         scheduler_cls = getattr(custom_schedulers, name)
#     except AttributeError:
#         scheduler_cls = getattr(torch.optim.lr_scheduler, name)
#     scheduler = scheduler_cls(optimizer, **kwargs)
#     return scheduler


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


@gin.configurable('neptune', blacklist=['save_path'])
def get_neptune(save_path: str, project_name: str, user_name: str, experiment_name: Optional[str] = None):
    from pytorch_lightning.loggers import NeptuneLogger
    api_token = os.environ["NEPTUNE_API_TOKEN"]
    description = os.path.basename(save_path)
    neptune = NeptuneLogger(api_key=api_token,
                            project_name=f'{user_name}/{project_name}',
                            experiment_name=experiment_name if experiment_name else description,
                            description=description)
    return neptune


def apply_neptune(callbacks: List[Callback], loggers: List[pl_loggers.LightningLoggerBase], *, save_path):
    neptune = get_neptune(save_path)
    loggers += [neptune]
    callbacks += [HyperparamsNeptuneSaver()]
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
                     num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    import tdc.single_pred

    task = getattr(tdc.single_pred, task_name)
    data = task(name=dataset_name)
    split = data.get_split(method=split_method, seed=split_seed, frac=split_frac)

    train_data = featurizer.encode_smiles_list(split['train']['Drug'], split['train']['Y'])
    valid_data = featurizer.encode_smiles_list(split['valid']['Drug'], split['valid']['Y'])
    test_data = featurizer.encode_smiles_list(split['test']['Drug'], split['test']['Y'])

    logging.info(f'Train samples: {len(train_data)}')
    logging.info(f'Validation samples: {len(valid_data)}')
    logging.info(f'Test samples: {len(test_data)}')

    train_loader, val_loader, test_loader = featurizer.get_data_loaders(train_data,
                                                                        valid_data,
                                                                        test_data,
                                                                        batch_size=batch_size,
                                                                        num_workers=num_workers)
    return train_loader, val_loader, test_loader


def evaluate_and_save_results(trainer: pl.Trainer, test_loader: DataLoader, save_path: str, evaluation: str):
    logging.info(f'Running test evaluation for {evaluation} weights')
    ckpt_path = 'best' if evaluation == 'best' else None
    results = trainer.test(test_dataloaders=test_loader, ckpt_path=ckpt_path)
    logging.info(results)
    with open(os.path.join(save_path, "test_results.json"), "w") as f:
        json.dump(results, f)
