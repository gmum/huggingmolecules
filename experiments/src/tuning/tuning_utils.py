import copy
import json
import os
from typing import Optional

from experiments.src.gin.gin_utils import bind_parameters_from_dict
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

    def __call__(self, trial: optuna.trial.Trial) -> float:
        trial_path = os.path.join(self.save_path, f'trial_{trial.number}')
        suggested_values = {p: getattr(trial, v[0])(p, *v[1:]) for p, v in self.optuna_params.items()}
        bind_parameters_from_dict(suggested_values)

        model = copy.deepcopy(self.model) if self.model else None
        trainer = train_model(model=model, featurizer=self.featurizer, root_path=trial_path)
        return trainer.logged_metrics[self.metric]


def enqueue_failed_trials(study, retry_not_completed: bool) -> None:
    from optuna.trial import TrialState
    params_set = set(tuple(trial.params) for trial in study.get_trials()
                     if trial.state == TrialState.COMPLETE)

    if retry_not_completed:
        condition = lambda trial: trial.state not in [TrialState.COMPLETE, TrialState.WAITING]
    else:
        condition = lambda trial: trial.state == TrialState.FAIL

    for trial in study.get_trials():
        if condition(trial) and tuple(trial.params) not in params_set:
            params_set.add(tuple(trial.params))
            study.enqueue_trial(trial.params)


def print_and_save_search_results(study, metric: str, save_path: str) -> None:
    dataframe = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    trial = study.best_trial

    print(str(dataframe))
    print(f'Best trial no {trial.number}')
    print(f' {metric}: {trial.value}')
    print(f' params:')
    for key, value in trial.params.items():
        print(f'  {key} = {value}')

    with open(os.path.join(save_path, 'trials.dataframe.txt'), 'w') as fp:
        fp.write(str(dataframe))
    with open(os.path.join(save_path, 'best_trial.json'), 'w') as fp:
        json.dump(trial.params, fp)
