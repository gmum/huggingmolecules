import gin

from experiments.training import tune_hyper
from experiments.training.training_benchmark_utils import print_benchmark_results
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase


def benchmark(model: PretrainedModelBase,
              featurizer: PretrainedFeaturizerMixin,
              prefix: str,
              results_only: bool = False):
    with gin.unlock_config():
        task_name = gin.query_parameter('data.task_name')
        dataset_name = gin.query_parameter('data.dataset_name')
        gin.bind_parameter('optuna.study_name', f'{prefix}_benchmark_{task_name}_{dataset_name}')

    if not results_only:
        tune_hyper(model, featurizer)

    print_benchmark_results()
