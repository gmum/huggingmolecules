import copy
import json
import logging
import os
from collections import defaultdict
from typing import Dict, Any, Optional, List

import gin

from experiments.src.training.training_train_model import train_model
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase


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
            if not param.startswith('ignore.'):
                gin.bind_parameter(param, value)


class Objective:
    import optuna

    def __init__(self,
                 model: Optional[PretrainedModelBase],
                 featurizer: Optional[PretrainedFeaturizerMixin],
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

        model = copy.deepcopy(self.model) if self.model else None
        trainer = train_model(model=model, featurizer=self.featurizer, root_path=trial_path)
        return trainer.logged_metrics[self.metric]


class WeightRemoverCallbackBase:
    def __init__(self, save_path: str):
        self.save_path = save_path

    def _get_paths_to_remove(self, study):
        raise NotImplementedError

    def __call__(self, study, trial):
        for trial_dir in self._get_paths_to_remove(study):
            path = os.path.join(self.save_path, trial_dir)
            cmd = f'rm -f {path}/*.ckpt'
            logging.info(f'Running command: {cmd}')
            os.system(cmd)


@gin.configurable('weight_remover')
def get_weight_remover(save_path, type: str = 'all', group_by: Optional[List[str]] = None):
    if type == 'all':
        return WeightRemoverAll(save_path)
    elif type == 'keep_best':
        return WeightRemoverKeepBest(save_path)
    elif type == 'group_by':
        if not group_by:
            raise ValueError("Please specify weight_remover.group_by in config.")
        direction = gin.query_parameter('optuna.direction')
        return WeightRemoverGroupBy(save_path, direction, group_by)


class WeightRemoverAll(WeightRemoverCallbackBase):
    def _get_paths_to_remove(self, study):
        from optuna.trial import TrialState
        return [f'trial_{t.number}' for t in study.get_trials() if t.state == TrialState.COMPLETE]


class WeightRemoverKeepBest(WeightRemoverCallbackBase):
    def _get_paths_to_remove(self, study):
        from optuna.trial import TrialState
        completed = [t for t in study.get_trials() if t.state == TrialState.COMPLETE]
        return [f'trial_{t.number}' for t in completed if t.value != study.best_trial.value]


class WeightRemoverGroupBy(WeightRemoverCallbackBase):
    def __init__(self, save_path: str, direction: str, group_by: List[str]):
        super().__init__(save_path)
        self.direction = direction
        self.group_by = group_by

    def _get_paths_to_remove(self, study):
        from optuna.trial import TrialState
        completed = [t for t in study.get_trials() if t.state == TrialState.COMPLETE]
        groups = self._get_params_group_dict(completed)
        return [f'trial_{t.number}' for t in completed if
                all(groups[p][t.params[p]] != t.value for p in self.group_by)]

    def _get_params_group_dict(self, completed):
        params_dict = {}
        agg = max if self.direction == 'maximize' else min
        for param_name in self.group_by:
            grouped = defaultdict(list)
            for t in completed:
                grouped[t.params[param_name]].append(t.value)
            params_dict[param_name] = {param: agg(values) for param, values in grouped.items()}
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
