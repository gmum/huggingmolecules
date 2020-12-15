from typing import Literal

from pytorch_lightning import Trainer

from .training_lightning_module import TrainingModule
from .training_train_model_utils import *
from .training_train_model_utils import get_custom_callbacks, apply_neptune, evaluate_and_save_results


@gin.configurable('train', blacklist=['model', 'featurizer'])
def train_model(*,
                model: Optional[PretrainedModelBase] = None,
                featurizer: Optional[PretrainedFeaturizerMixin] = None,
                save_path: str,
                num_epochs: int,
                gpus: List[int],
                resume: bool = False,
                use_neptune: bool = False,
                evaluation: Optional[str] = None,
                custom_callbacks: Optional[List[str]] = None,
                task: Literal["regression", "classification"] = "regression"):
    if not model:
        model, featurizer = get_model_and_featurizer(task=task)

    resume_path = os.path.join(save_path, 'last.ckpt')
    if not resume and os.path.exists(resume_path):
        raise IOError(f'Please clear {save_path} folder before running or pass train.resume=True')

    callbacks = get_default_callbacks(save_path)
    callbacks += get_custom_callbacks(custom_callbacks)
    loggers = get_default_loggers(save_path)

    if use_neptune:
        apply_neptune(model, callbacks, loggers, save_path=save_path)

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
    pl_module = TrainingModule(model, optimizer=optimizer, loss_fn=loss_fn, task=task)

    train_loader, val_loader, test_loader = get_data_loaders(featurizer)

    trainer.fit(pl_module,
                train_dataloader=train_loader,
                val_dataloaders=val_loader)

    if evaluation:
        evaluate_and_save_results(trainer, test_loader, save_path, evaluation)

    return trainer
