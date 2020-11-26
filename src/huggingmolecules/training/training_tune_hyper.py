from typing import Optional

import gin

from .training_tune_hyper_utils import print_and_save_info, get_sampler, Objective, WeightRemoverCallback, \
    enqueue_failed_trials
from ..featurization.featurization_api import PretrainedFeaturizerMixin
from ..models.models_api import PretrainedModelBase


@gin.configurable('optuna', blacklist=['model', 'featurizer'])
def tune_hyper(model: PretrainedModelBase,
               featurizer: PretrainedFeaturizerMixin, *,
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
               keep_best_only: bool = False):
    import optuna

    sampler = get_sampler(sampler_name, params)

    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=resume,
                                sampler=sampler,
                                direction=direction)

    if resume:
        enqueue_failed_trials(study)

    objective = Objective(model,
                          featurizer,
                          save_path=save_path,
                          optuna_params=params,
                          metric=metric)

    callbacks = [WeightRemoverCallback(save_path, direction)] if keep_best_only else None
    study.optimize(objective,
                   n_trials=n_trials,
                   timeout=timeout,
                   callbacks=callbacks)

    print_and_save_info(study, save_path, metric)
