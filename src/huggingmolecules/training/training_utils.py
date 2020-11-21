import os
from typing import Tuple, List

import gin
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader

import src.huggingmolecules.training.training_callbacks as custom_callbacks
from src.huggingmolecules import split_data_random


def get_default_loggers(save_path):
    return [pl_loggers.CSVLogger(save_path)]


def get_default_callbacks(save_path):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(save_path, "weights"),
        verbose=True,
        save_last=True,
        monitor='valid_loss',
        mode='min')
    return [checkpoint_callback, custom_callbacks.Heartbeat(), custom_callbacks.MetaSaver()]


def get_custom_callbacks(callbacks_names: List[str] = None):
    callbacks_names = callbacks_names if callbacks_names else []
    callbacks = []
    for clb_name in callbacks_names:
        clb_cls = getattr(custom_callbacks, clb_name)
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
def get_loss_fn(name='mse_loss', **kwargs):
    return getattr(F, name)


@gin.configurable('optimizer', blacklist=['model'])
def get_optimizer(model, name='Adam', **kwargs):
    opt_cls = getattr(torch.optim, name)
    return opt_cls(model.parameters(), **kwargs)


@gin.configurable('neptune', blacklist=['save_path'])
def get_neptune(save_path, project_name, user_name, experiment_name=None):
    from pytorch_lightning.loggers import NeptuneLogger
    api_token = os.environ["NEPTUNE_API_TOKEN"]
    description = os.path.basename(save_path)
    neptune = NeptuneLogger(api_key=api_token,
                            project_name=f'{user_name}/{project_name}',
                            experiment_name=experiment_name if experiment_name else description,
                            description=description)
    return neptune


@gin.configurable('data', blacklist=['featurizer', 'batch_size'])
def get_data_loaders(featurizer, *, data_path, batch_size, train_size=0.9, test_size=0.0, num_workers=1,
                     split_path=None, seed=None) -> \
        Tuple[DataLoader, DataLoader, DataLoader]:
    if split_path and seed:
        raise ValueError("At least one of the 'split_path' and 'seed' parameters must be set to None")

    dataset = featurizer.load_dataset_from_csv(data_path)
    if split_path:
        train_data, val_data, test_data = split_data_random(dataset, train_size, test_size, seed)
    else:
        train_data, val_data, test_data = split_data_random(dataset, train_size, test_size, seed)

    train_loader, val_loader, test_loader = featurizer.get_data_loaders(train_data, val_data, test_data,
                                                                        batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader
