from typing import Optional

from pytorch_lightning import Trainer

from .training_lightning_module import TrainingModule
from .training_utils import *
from .training_utils import get_custom_callbacks, _apply_neptune, evaluate_and_save_results
from ..gin import get_default_name


@gin.configurable('train', blacklist=['model', 'featurizer'])
def train_model(*,
                model: Optional[PretrainedModelBase] = None,
                featurizer: Optional[PretrainedFeaturizerMixin] = None,
                root_path: str,
                num_epochs: int,
                gpus: List[int],
                resume: bool = False,
                use_neptune: bool = False,
                evaluation: Optional[str] = None,
                custom_callbacks: Optional[List[str]] = None,
                batch_size: int,
                num_workers: int = 0,
                cache_encodings: bool = True):
    if not model:
        gin_model = GinModel()
        model = gin_model.produce_model()
        featurizer = gin_model.produce_featurizer()

    study_name = get_default_name()
    save_path = os.path.join(root_path, study_name)

    resume_path = os.path.join(save_path, 'last.ckpt')
    if not resume and os.path.exists(resume_path):
        raise IOError(f'Please clear {save_path} folder before running or pass train.resume=True')

    callbacks = get_default_callbacks(save_path)
    callbacks += get_custom_callbacks(custom_callbacks)
    loggers = get_default_loggers(save_path)

    if use_neptune:
        _apply_neptune(model, callbacks, loggers, neptune_experiment_name=study_name, neptune_description=save_path)

    trainer = Trainer(default_root_dir=save_path,
                      max_epochs=num_epochs,
                      callbacks=callbacks,
                      logger=loggers,
                      log_every_n_steps=1,
                      checkpoint_callback=True,
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

    if evaluation:
        evaluate_and_save_results(trainer, test_loader, save_path, evaluation)

    return trainer
