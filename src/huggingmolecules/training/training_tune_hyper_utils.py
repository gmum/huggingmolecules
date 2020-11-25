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


@gin.configurable('optuna.keep_best_only', blacklist=['save_path', 'direction'])
def get_remove_all_weights_except_best(save_path: str,
                                       direction: str,
                                       enabled: bool = False,
                                       group_by: Optional[List[str]] = None):
    import optuna
    from optuna.trial import TrialState
    if not enabled:
        return None

    if group_by:
        def get_params_group_dict(completed):
            params_dict = {}
            for param_name in group_by:
                grouped = defaultdict(list)
                for t in completed:
                    grouped[t.params[param_name]].append(t.value)
                if direction == 'maximize':
                    params_dict[param_name] = {param: max(values) for param, values in grouped.items()}
                else:
                    params_dict[param_name] = {param: min(values) for param, values in grouped.items()}
            return params_dict

        def get_best_paths(study):
            completed = [t for t in study.get_trials() if t.state == TrialState.COMPLETE]
            groups = get_params_group_dict(completed)
            best_paths = []
            for t in completed:
                if any(groups[p][t.params[p]] == t.value for p in group_by):
                    best_paths.append(f'trial_{t.number}')

            return best_paths
    else:
        def get_best_paths(study):
            return [f'trial_{study.best_trial.number}']

    def remove_all_weights_except_best(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        best_paths = get_best_paths(study)
        for trial_dir in os.listdir(save_path):
            if trial_dir not in best_paths:
                path = os.path.join(save_path, trial_dir)
                cmd = f'rm -f {path}/*.ckpt'
                logging.info(f'Running command: {cmd}')
                os.system(cmd)

    return remove_all_weights_except_best


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
