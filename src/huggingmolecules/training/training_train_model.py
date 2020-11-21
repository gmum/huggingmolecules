import json

from pytorch_lightning import Trainer

from .training_lightning_module import TrainingModule
from .training_utils import *
from .training_utils import get_custom_callbacks, apply_neptune
from ..featurization.featurization_api import PretrainedFeaturizerMixin
from ..models.models_api import PretrainedModelBase


@gin.configurable('train', blacklist=['model', 'featurizer'])
def train_model(model: PretrainedModelBase,
                featurizer: PretrainedFeaturizerMixin, *,
                save_path: str,
                num_epochs: int,
                batch_size: int,
                gpus: List[int],
                resume: bool = False,
                use_neptune: bool = False,
                evaluate: bool = False,
                custom_callbacks: Optional[List[str]] = None):
    resume_path = os.path.join(save_path, 'last.ckpt')
    if not resume and os.path.exists(resume_path):
        raise IOError(f'Please clear {save_path} folder before running or pass train.resume=True')

    callbacks = get_default_callbacks(save_path)
    callbacks += get_custom_callbacks(custom_callbacks)
    loggers = get_default_loggers(save_path)

    if use_neptune:
        apply_neptune(callbacks, loggers, save_path=save_path)

    trainer = Trainer(default_root_dir=save_path,
                      max_epochs=num_epochs,
                      callbacks=callbacks,
                      logger=loggers,
                      log_every_n_steps=1,
                      checkpoint_callback=True,
                      resume_from_checkpoint=resume_path if resume else None,
                      gpus=gpus)

    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()
    pl_module = TrainingModule(model, optimizer=optimizer, loss_fn=loss_fn)

    train_loader, val_loader, test_loader = get_data_loaders(featurizer, batch_size=batch_size)

    trainer.fit(pl_module,
                train_dataloader=train_loader,
                val_dataloaders=val_loader)

    if evaluate:
        results = trainer.test(test_dataloaders=test_loader)
        logging.info(results)
        with open(os.path.join(save_path, "test_results.json"), "w") as f:
            json.dump(results, f)

    return trainer
