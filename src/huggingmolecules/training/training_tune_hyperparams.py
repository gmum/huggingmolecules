import copy
import os
import shutil
from typing import Optional

import gin

from . import TrainingModule
from .training_train_model import train_model, get_optimizer, get_loss_fn
from ..featurization.featurization_api import PretrainedFeaturizerMixin
from ..models.models_api import PretrainedModelBase


def get_objective(model: PretrainedModelBase, featurizer: PretrainedFeaturizerMixin, *, save_path: str,
                  optimizer_optuna_params: dict, metrics: str):
    def objective(trial):
        trial_path = os.path.join(save_path, f'trial_{trial.number}')
        optimizer_params = {p: getattr(trial, v[0])(p, *v[1:])
                            for p, v in optimizer_optuna_params.items()}

        optimizer = get_optimizer(model, **optimizer_params)
        loss_fn = get_loss_fn()
        pl_module = TrainingModule(model, loss_fn=loss_fn, optimizer=optimizer)
        trainer = train_model(pl_module, featurizer, save_path=trial_path)
        return trainer.logged_metrics[metrics]

    return objective


@gin.configurable('optuna', blacklist=['model', 'featurizer'])
def tune_hyperparams(model: PretrainedModelBase, featurizer: PretrainedFeaturizerMixin, save_path: str,
                     optimizer_params: dict, direction: str, metrics: str, n_trials: Optional[int] = None,
                     timeout: Optional[float] = None, sampler_name: str = 'TPESampler'):
    import optuna
    sampler_cls = getattr(optuna.samplers, sampler_name)
    if sampler_cls is optuna.samplers.GridSampler:
        search_grid = copy.deepcopy(optimizer_params)
        optimizer_params = {k: ('suggest_uniform', 0, 1) for k in optimizer_params.keys()}
        sampler = sampler_cls(search_grid)
    else:
        sampler = sampler_cls()

    objective = get_objective(model, featurizer,
                              save_path=save_path,
                              optimizer_optuna_params=optimizer_params,
                              metrics=metrics)

    study = optuna.create_study(sampler=sampler, direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(save_path)
