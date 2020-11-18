import json
import logging
import os

import gin
import torch
from pytorch_lightning import Trainer
import pytorch_lightning as pl

from .training_lightning_module import TrainingModule
from .training_utils import *

logger = logging.getLogger(__name__)


@gin.configurable('train', blacklist=['model', 'featurizer'])
def train_model(model, featurizer, *, save_path, num_epochs, batch_size, gpus, resume=False, use_neptune=False,
                evaluate=False):
    resume_path = os.path.join(save_path, 'last.ckpt')
    if not resume and os.path.exists(resume_path):
        raise IOError(f"Please clear {save_path} folder before running or pass train.resume=True")

    train_loader, val_loader, test_loader = get_data_loaders(featurizer, batch_size=batch_size)

    loggers = default_loggers(save_path)
    callbacks = default_callbacks(save_path)
    if use_neptune:
        loggers, callbacks = apply_neptune(loggers, callbacks)

    trainer = Trainer(
        default_root_dir=save_path,
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        checkpoint_callback=True,
        resume_from_checkpoint=resume_path if resume else None,
        gpus=gpus if torch.cuda.is_available() else 0)  # TODO remove if

    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()
    pl_module = TrainingModule(model, optimizer=optimizer, loss_fn=loss_fn)
    # trainer.fit(pl_module, train_dataloader=train_loader, val_dataloaders=val_loader)

    if evaluate:
        results, = trainer.test(test_dataloaders=test_loader)
        logger.info(results)
        with open(os.path.join(save_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

    return trainer
