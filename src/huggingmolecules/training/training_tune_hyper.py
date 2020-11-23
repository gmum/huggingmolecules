from typing import Optional

import gin

from .training_tune_hyper_utils import print_and_save_info, get_sampler, get_objective, \
    get_remove_all_weights_except_best
from ..featurization.featurization_api import PretrainedFeaturizerMixin
from ..models.models_api import PretrainedModelBase


@gin.configurable('optuna', blacklist=['model', 'featurizer'])
def tune_hyper(model: PretrainedModelBase,
               featurizer: PretrainedFeaturizerMixin,
               save_path: str,
               params: dict,
               direction: str,
               metric: str,
               n_trials: Optional[int] = None,
               timeout: Optional[float] = None,
               sampler_name: str = 'TPESampler',
               study_name: str = None,
               storage: Optional[str] = None,
               resume: bool = False,
               keep_best_weights_only: bool = False):
    import optuna

    sampler = get_sampler(sampler_name, params)

    objective = get_objective(model,
                              featurizer,
                              save_path=save_path,
                              optuna_params=params,
                              metric=metric)

    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=resume,
                                sampler=sampler,
                                direction=direction)

    if resume:
        added_params = set()
        for trial in study.get_trials():
            if trial.state == optuna.trial.TrialState.FAIL and tuple(trial.params) not in added_params:
                added_params.add(tuple(trial.params))
                study.enqueue_trial(trial.params)

    callbacks = [get_remove_all_weights_except_best(save_path)] if keep_best_weights_only else None

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)

    print_and_save_info(study, save_path, metric)
