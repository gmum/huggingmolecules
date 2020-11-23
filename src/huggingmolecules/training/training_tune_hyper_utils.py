import copy
import json
import logging
import os
from typing import Dict, Any

import gin

from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase
from src.huggingmolecules.training import train_model


def get_sampler(name: str, params: dict):
    import optuna
    sampler_cls = getattr(optuna.samplers, name)
    if sampler_cls is optuna.samplers.GridSampler:
        search_grid = copy.deepcopy(params)
        for k in params.keys():
            params[k] = ('suggest_uniform', 0, 1)
        return sampler_cls(search_grid)
    else:
        return sampler_cls()


def get_objective(model: PretrainedModelBase,
                  featurizer: PretrainedFeaturizerMixin, *,
                  save_path: str,
                  optuna_params: dict,
                  metric: str):
    def objective(trial):
        trial_path = os.path.join(save_path, f'trial_{trial.number}')
        suggested_values = {p: getattr(trial, v[0])(p, *v[1:]) for p, v in optuna_params.items()}
        bind_parameters_from_dict(suggested_values)
        model_copy = copy.deepcopy(model)
        trainer = train_model(model_copy, featurizer, save_path=trial_path)
        return trainer.logged_metrics[metric]

    return objective


def bind_parameters_from_dict(values_dict: Dict[str, Any]):
    with gin.unlock_config():
        for param, value in values_dict.items():
            gin.bind_parameter(param, value)


def get_remove_all_weights_except_best(save_path):
    import optuna

    def remove_all_weights_except_best(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        best_trial_dir = f'trial_{study.best_trial.number}'
        for trial_dir in os.listdir(save_path):
            if trial_dir != best_trial_dir:
                path = os.path.join(save_path, trial_dir)
                cmd = f'rm {path}/*.ckpt'
                logging.info(f'Running command: {cmd}')
                os.system(cmd)

    return remove_all_weights_except_best


def print_and_save_info(study, save_path: str, metric: str):
    dataframe = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(str(dataframe))
    with open(os.path.join(save_path, 'trials.dataframe.json'), 'w') as fp:
        fp.write(str(dataframe))

    trial = study.best_trial
    print(f'Best trial no {trial.number}')
    print(f' {metric}: {trial.value}')
    print(f' params:')
    for key, value in trial.params.items():
        print(f'  {key} = {value}')
    with open(os.path.join(save_path, 'best_trial.json'), 'w') as fp:
        json.dump(trial.params, fp)
