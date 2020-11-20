import os
from typing import Tuple

import gin
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.huggingmolecules import split_data_random
from src.huggingmolecules.callbacks import Heartbeat, MetaSaver
import src.huggingmolecules.callbacks.callbacks_schedulers as custom_schedulers


def default_loggers(save_path):
    return [pl_loggers.CSVLogger(save_path)]


def default_callbacks(save_path):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(save_path, "weights"),
        verbose=True,
        save_last=True,
        monitor='valid_loss',
        mode='min')
    return [checkpoint_callback, Heartbeat(), MetaSaver(), LearningRateMonitor(logging_interval='step')]


@gin.configurable('lr_scheduler', blacklist=['optimizer'])
def get_lr_scheduler(optimizer, name='IdentityScheduler', **kwargs):
    try:
        scheduler_cls = getattr(custom_schedulers, name)
    except AttributeError:
        scheduler_cls = getattr(torch.optim.lr_scheduler, name)
    scheduler = scheduler_cls(optimizer, **kwargs)
    return scheduler


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
    neptune = NeptuneLogger(api_key=api_token,
                            project_name=f'{user_name}/{project_name}',
                            experiment_name=experiment_name if experiment_name else os.path.basename(save_path))
    return neptune


@gin.configurable('data', blacklist=['featurizer', 'batch_size'])
def get_data_loaders(featurizer, *, data_path, batch_size, seed=None, train_size=0.9, test_size=0.0, num_workers=1) -> \
        Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = featurizer.load_dataset_from_csv(data_path)
    train_data, val_data, test_data = split_data_random(dataset, train_size, test_size, seed)
    train_loader, val_loader, test_loader = featurizer.get_data_loaders(train_data, val_data, test_data,
                                                                        batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader
