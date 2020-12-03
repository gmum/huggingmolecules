from experiments.training import tune_hyper
from experiments.training.training_benchmark_utils import print_benchmark_results, set_default_experiment_name
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase


def benchmark(model: PretrainedModelBase,
              featurizer: PretrainedFeaturizerMixin,
              prefix: str):
    set_default_experiment_name(prefix)

    tune_hyper(model, featurizer)

    print_benchmark_results()
