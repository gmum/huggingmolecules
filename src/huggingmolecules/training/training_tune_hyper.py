import copy
import json
import os
from typing import Optional, Any, Dict

import gin

from .training_train_model import train_model
from ..featurization.featurization_api import PretrainedFeaturizerMixin
from ..models.models_api import PretrainedModelBase


def bind_parameters_from_dict(values_dict: Dict[str, Any]):
    with gin.unlock_config():
        for param, value in values_dict.items():
            gin.bind_parameter(param, value)


def get_objective(model: PretrainedModelBase, featurizer: PretrainedFeaturizerMixin, *, save_path: str,
                  optuna_params: dict, metrics: str):
    def objective(trial):
        trial_path = os.path.join(save_path, f'trial_{trial.number}')
        suggested_values = {p: getattr(trial, v[0])(p, *v[1:]) for p, v in optuna_params.items()}
        bind_parameters_from_dict(suggested_values)
        model_copy = copy.deepcopy(model)
        trainer = train_model(model_copy, featurizer, save_path=trial_path)
        return trainer.logged_metrics[metrics]

    return objective


@gin.configurable('optuna', blacklist=['model', 'featurizer'])
def tune_hyper(model: PretrainedModelBase, featurizer: PretrainedFeaturizerMixin, save_path: str,
               params: dict, direction: str, metrics: str, n_trials: Optional[int] = None,
               timeout: Optional[float] = None, sampler_name: str = 'TPESampler', study_name: str = None):
    import optuna
    sampler_cls = getattr(optuna.samplers, sampler_name)
    if sampler_cls is optuna.samplers.GridSampler:
        search_grid = copy.deepcopy(params)
        params = {k: ('suggest_uniform', 0, 1) for k in params.keys()}
        sampler = sampler_cls(search_grid)
    else:
        sampler = sampler_cls()

    objective = get_objective(model, featurizer,
                              save_path=save_path,
                              optuna_params=params,
                              metrics=metrics)

    study = optuna.create_study(study_name=study_name, sampler=sampler, direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    trial = study.best_trial

    with open(os.path.join(save_path, 'best_trial.json'), 'w') as fp:
        json.dump(trial.params, fp)

    print(f'Best trial no {trial.number}')
    print(f' {metrics}: {trial.value}')
    print(f' params:')
    for key, value in trial.params.items():
        print(f'  {key} = {value}')
