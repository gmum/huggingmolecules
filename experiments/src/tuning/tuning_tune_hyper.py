import os
from typing import Optional

import gin

from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase
from .tuning_utils import print_and_save_info, get_sampler, Objective, \
    enqueue_failed_trials, get_weight_remover
from ..gin import get_default_name


@gin.configurable('optuna', blacklist=['model', 'featurizer'])
def tune_hyper(*,
               model: Optional[PretrainedModelBase] = None,
               featurizer: Optional[PretrainedFeaturizerMixin] = None,
               root_path: str,
               params: dict,
               direction: str,
               metric: str,
               n_trials: Optional[int] = None,
               timeout: Optional[float] = None,
               sampler_name: str = 'TPESampler',
               storage: Optional[str] = None,
               resume: bool = False,
               weight_remover: Optional[str] = None):
    import optuna

    study_name = get_default_name()

    sampler = get_sampler(sampler_name, params)

    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=resume,
                                sampler=sampler,
                                direction=direction)

    if resume:
        enqueue_failed_trials(study)

    save_path = os.path.join(root_path, study_name)
    objective = Objective(model,
                          featurizer,
                          save_path=save_path,
                          optuna_params=params,
                          metric=metric)

    callbacks = [get_weight_remover(save_path)] if weight_remover else None
    study.optimize(objective,
                   n_trials=n_trials,
                   timeout=timeout,
                   callbacks=callbacks)

    print_and_save_info(study, save_path, metric)
