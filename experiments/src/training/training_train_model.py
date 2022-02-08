from typing import Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .training_lightning_module import TrainingModule
from .training_utils import *
from .training_utils import get_custom_callbacks, evaluate_and_save_results, get_neptune_logger, apply_neptune_logger
from ..gin.gin_utils import get_default_experiment_name


@gin.configurable('train', denylist=['model', 'featurizer'])
def train_model(*,
                model: Optional[PretrainedModelBase] = None,
                featurizer: Optional[PretrainedFeaturizerMixin] = None,
                root_path: str,
                num_epochs: int,
                gpus: List[int],
                resume: bool = False,
                save_checkpoints: bool = True,
                use_neptune: bool = False,
                evaluate: bool = True,
                custom_callbacks: Optional[List[str]] = None,
                batch_size: int,
                num_workers: int = 0,
                cache_encodings: bool = True):
    if not model:
        gin_model = GinModel()
        model = gin_model.produce_model()
        featurizer = gin_model.produce_featurizer()

    study_name = get_default_experiment_name()
    save_path = os.path.join(root_path, study_name)

    resume_path = os.path.join(save_path, 'last.ckpt')
    if not resume and os.path.exists(resume_path):
        raise IOError(f'Please clear {save_path} folder before running or pass train.resume=True')

    callbacks = get_default_callbacks() + get_custom_callbacks(custom_callbacks)
    loggers = get_default_loggers(save_path)
    if save_checkpoints:
        callbacks += [ModelCheckpoint(dirpath=save_path, save_last=True)]
    if use_neptune:
        neptune_logger = get_neptune_logger(model, experiment_name=study_name, description=save_path)
        apply_neptune_logger(neptune_logger, callbacks, loggers)

    trainer = Trainer(default_root_dir=save_path,
                      max_epochs=num_epochs,
                      callbacks=callbacks,
                      checkpoint_callback=save_checkpoints,
                      logger=loggers,
                      log_every_n_steps=1,
                      resume_from_checkpoint=resume_path if resume else None,
                      gpus=gpus)

    pl_module = TrainingModule(model,
                               optimizer=get_optimizer(model),
                               loss_fn=get_loss_fn(),
                               metric_cls=get_metric_cls())

    train_loader, val_loader, test_loader = get_data_loaders(featurizer,
                                                             batch_size=batch_size,
                                                             num_workers=num_workers,
                                                             cache_encodings=cache_encodings)

    trainer.fit(pl_module,
                train_dataloader=train_loader,
                val_dataloaders=val_loader)

    if evaluate:
        results_path = os.path.join(save_path, 'results.json')
        evaluate_and_save_results(trainer, test_loader, results_path)
        if use_neptune:
            neptune_logger.log_artifact(results_path)

    return trainer
