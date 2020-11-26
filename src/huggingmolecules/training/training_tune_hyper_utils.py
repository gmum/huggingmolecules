import copy
import json
import logging
import os
from collections import defaultdict
from typing import Dict, Any, Optional, List

import gin

from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase
from src.huggingmolecules.training import train_model


def get_sampler(name: str, params: dict):
    import optuna
    sampler_cls = getattr(optuna.samplers, name)
    if sampler_cls is optuna.samplers.GridSampler:
        search_grid = copy.deepcopy(params)
        for k, v in params.items():
            params[k] = ('suggest_categorical', v)
        return sampler_cls(search_grid)
    else:
        return sampler_cls()


def bind_parameters_from_dict(values_dict: Dict[str, Any]):
    with gin.unlock_config():
        for param, value in values_dict.items():
            gin.bind_parameter(param, value)


class Objective:
    import optuna

    def __init__(self,
                 model: PretrainedModelBase,
                 featurizer: PretrainedFeaturizerMixin,
                 save_path: str,
                 optuna_params: dict,
                 metric: str):
        self.model = model
        self.featurizer = featurizer
        self.save_path = save_path
        self.optuna_params = optuna_params
        self.metric = metric

    def __call__(self, trial: optuna.trial.Trial):
        trial_path = os.path.join(self.save_path, f'trial_{trial.number}')
        suggested_values = {p: getattr(trial, v[0])(p, *v[1:]) for p, v in self.optuna_params.items()}
        bind_parameters_from_dict(suggested_values)
        model_copy = copy.deepcopy(self.model)
        trainer = train_model(model_copy, self.featurizer, save_path=trial_path)
        return trainer.logged_metrics[self.metric]


@gin.configurable('optuna.keep_best_only', blacklist=['save_path', 'direction'])
class WeightRemoverCallback:
    def __init__(self,
                 save_path: str,
                 direction: str,
                 group_by: Optional[List[str]] = None):
        self.save_path = save_path
        self.direction = direction
        self.group_by = group_by

    def __call__(self, study, trial):
        for trial_dir in self._get_paths_to_remove(study):
            path = os.path.join(self.save_path, trial_dir)
            cmd = f'rm -f {path}/*.ckpt'
            logging.info(f'Running command: {cmd}')
            os.system(cmd)

    def _get_paths_to_remove(self, study):
        from optuna.trial import TrialState
        completed = [t for t in study.get_trials() if t.state == TrialState.COMPLETE]
        if self.group_by:
            groups = self._get_params_group_dict(completed)
            return [f'trial_{t.number}' for t in completed if
                    all(groups[p][t.params[p]] != t.value for p in self.group_by)]
        else:
            return [f'trial_{t.number}' for t in completed if t.value != study.best_trial.value]

    def _get_params_group_dict(self, completed):
        params_dict = {}
        for param_name in self.group_by:
            grouped = defaultdict(list)
            for t in completed:
                grouped[t.params[param_name]].append(t.value)
            if self.direction == 'maximize':
                params_dict[param_name] = {param: max(values) for param, values in grouped.items()}
            else:
                params_dict[param_name] = {param: min(values) for param, values in grouped.items()}
        return params_dict


def enqueue_failed_trials(study):
    import optuna
    params_set = set(tuple(trial.params) for trial in study.get_trials()
                     if trial.state == optuna.trial.TrialState.COMPLETE)
    for trial in study.get_trials():
        if trial.state == optuna.trial.TrialState.FAIL and tuple(trial.params) not in params_set:
            params_set.add(tuple(trial.params))
            study.enqueue_trial(trial.params)


def print_and_save_info(study, save_path: str, metric: str):
    dataframe = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(str(dataframe))
    with open(os.path.join(save_path, 'trials.dataframe.txt'), 'w') as fp:
        fp.write(str(dataframe))

    trial = study.best_trial
    print(f'Best trial no {trial.number}')
    print(f' {metric}: {trial.value}')
    print(f' params:')
    for key, value in trial.params.items():
        print(f'  {key} = {value}')
    with open(os.path.join(save_path, 'best_trial.json'), 'w') as fp:
        json.dump(trial.params, fp)
