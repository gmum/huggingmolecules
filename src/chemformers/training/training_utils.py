import os

import gin
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.chemformers import split_data_random
from src.chemformers.callbacks import Heartbeat, MetaSaver
from torch.nn import functional as F


def default_loggers(save_path):
    return [pl_loggers.CSVLogger(save_path)]


def default_callbacks(save_path):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(save_path, "weights"),
        verbose=True,
        save_last=True,
        monitor='valid_loss',
        mode='min')
    return [checkpoint_callback]


@gin.configurable('loss_fn')
def get_loss_fn(name='mse_loss', **kwargs):
    return getattr(F, name)


@gin.configurable('optimizer', blacklist=['model'])
def get_optimizer(model, name='Adam', **kwargs):
    opt_cls = getattr(torch.optim, name)
    return opt_cls(model.parameters(), **kwargs)


@gin.configurable('neptune', blacklist=['loggers', 'callbacks'])
def apply_neptune(loggers, callbacks, project_name, user_name, experiment_name):
    from pytorch_lightning.loggers import NeptuneLogger
    api_token = os.environ["NEPTUNE_API_TOKEN"]
    neptune = NeptuneLogger(api_key=api_token,
                            project_name=f'{user_name}/{project_name}',
                            experiment_name=experiment_name)
    loggers += [neptune]
    callbacks += [Heartbeat(), MetaSaver(), LearningRateMonitor()]
    return loggers, callbacks


@gin.configurable('data', blacklist=['featurizer', 'batch_size'])
def get_data_loaders(featurizer, *, data_path, batch_size, seed=None, train_size=0.9, test_size=0.0, num_workers=1):
    dataset = featurizer.load_dataset_from_csv(data_path)
    train_data, val_data, test_data = split_data_random(dataset, train_size, test_size, seed)
    train_loader, val_loader, test_loader = featurizer.get_data_loaders(train_data, val_data, test_data,
                                                                        batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader
