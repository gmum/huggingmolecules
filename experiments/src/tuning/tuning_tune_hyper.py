import os
from typing import Optional

import gin

from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase
from .tuning_utils import get_sampler, Objective, \
    enqueue_failed_trials, print_and_save_search_results
from ..gin.gin_utils import get_default_experiment_name


@gin.configurable('optuna', denylist=['model', 'featurizer'])
def tune_hyper(*,
               model: Optional[PretrainedModelBase] = None,
               featurizer: Optional[PretrainedFeaturizerMixin] = None,
               root_path: str,
               params: dict,
               direction: str = 'minimize',
               metric: str = 'valid_loss',
               n_trials: Optional[int] = None,
               timeout: Optional[float] = None,
               sampler_name: str = 'TPESampler',
               storage: Optional[str] = None,
               resume: bool = False,
               retry_not_completed: bool = False,
               print_and_save_results: bool = True):
    import optuna

    study_name = get_default_experiment_name()

    sampler = get_sampler(sampler_name, params)

    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=resume,
                                sampler=sampler,
                                direction=direction)

    if resume:
        enqueue_failed_trials(study, retry_not_completed)

    save_path = os.path.join(root_path, study_name)
    objective = Objective(model,
                          featurizer,
                          save_path=save_path,
                          optuna_params=params,
                          metric=metric)

    study.optimize(objective,
                   n_trials=n_trials,
                   timeout=timeout)

    if print_and_save_results:
        print_and_save_search_results(study, metric, save_path)
