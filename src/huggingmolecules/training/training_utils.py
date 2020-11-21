import logging
import os
from typing import Tuple, List, Optional

import gin
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader

import src.huggingmolecules.training.training_callbacks as custom_callbacks_module
from src.huggingmolecules import split_data_random
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.featurization.featurization_utils import split_data_from_file
from src.huggingmolecules.training.training_callbacks import NeptuneCompatibleCallback, HyperparamsNeptuneSaver, \
    GinConfigSaver, ModelConfigSaver


def get_default_loggers(save_path: str) -> List[pl_loggers.LightningLoggerBase]:
    return [pl_loggers.CSVLogger(save_path)]


def get_default_callbacks(save_path: str) -> List[Callback]:
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(save_path, "weights"),
                                          verbose=True,
                                          save_last=True,
                                          monitor='valid_loss',
                                          mode='min')
    return [checkpoint_callback, GinConfigSaver(), ModelConfigSaver()]


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


@gin.configurable('data', blacklist=['featurizer', 'batch_size'])
def get_data_loaders(featurizer: PretrainedFeaturizerMixin, *,
                     data_path: str,
                     batch_size: int,
                     train_size: float = 0.9,
                     test_size: float = 0.0,
                     num_workers: int = 1,
                     split_path: Optional[str] = None,
                     seed: Optional[int] = None,
                     cache: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = featurizer.load_dataset_from_csv(data_path, cache=cache)
    if split_path:
        train_data, val_data, test_data = split_data_from_file(dataset, split_path)
    else:
        train_data, val_data, test_data = split_data_random(dataset, train_size, test_size, seed)

    logging.info(f'Train samples: {len(train_data)}')
    logging.info(f'Validation samples: {len(val_data)}')
    logging.info(f'Test samples: {len(test_data)}')

    train_loader, val_loader, test_loader = featurizer.get_data_loaders(train_data,
                                                                        val_data,
                                                                        test_data,
                                                                        batch_size=batch_size,
                                                                        num_workers=num_workers)
    return train_loader, val_loader, test_loader


def apply_neptune(callbacks: List[Callback], loggers: List[pl_loggers.LightningLoggerBase], *, save_path):
    neptune = get_neptune(save_path)
    loggers += [neptune]
    callbacks += [HyperparamsNeptuneSaver()]
    for clb in callbacks:
        if isinstance(clb, NeptuneCompatibleCallback):
            clb.neptune = neptune
