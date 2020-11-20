import json
import logging

from pytorch_lightning import Trainer

from .training_lightning_module import TrainingModule
from .training_utils import *
from ..models.models_api import PretrainedModelBase
from ..utils import parse_gin_str

logger = logging.getLogger(__name__)


@gin.configurable('train', blacklist=['model', 'featurizer'])
def train_model(model: PretrainedModelBase, featurizer, *, save_path, num_epochs, batch_size,
                gpus, resume=False, use_neptune=False,
                evaluate=False):
    os.makedirs(save_path, exist_ok=True)
    resume_path = os.path.join(save_path, 'last.ckpt')
    if not resume and os.path.exists(resume_path):
        raise IOError(f'Please clear {save_path} folder before running or pass train.resume=True')

    model_config_path = os.path.join(save_path, "model_config.json")
    model.get_config().save(model_config_path)

    gin_config_path = os.path.join(save_path, "gin_config.txt")
    gin_str = gin.config_str()
    with open(gin_config_path, "w") as f:
        f.write(gin_str)

    loggers = default_loggers(save_path)
    callbacks = default_callbacks(save_path)
    if use_neptune:
        neptune = get_neptune(save_path)
        neptune.log_artifact(model_config_path)
        neptune.log_artifact(gin_config_path)
        neptune.log_hyperparams(parse_gin_str(gin_str))
        loggers += [neptune]

    trainer = Trainer(
        default_root_dir=save_path,
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        checkpoint_callback=True,
        resume_from_checkpoint=resume_path if resume else None,
        gpus=gpus if torch.cuda.is_available() else 0)  # TODO remove if

    pl_module = TrainingModule(model,
                               optimizer=get_optimizer(model),
                               loss_fn=get_loss_fn())

    train_loader, val_loader, test_loader = get_data_loaders(featurizer, batch_size=batch_size)
    trainer.fit(pl_module, train_dataloader=train_loader, val_dataloaders=val_loader)

    if evaluate:
        results, = trainer.test(test_dataloaders=test_loader)
        logger.info(results)
        with open(os.path.join(save_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

    return trainer
