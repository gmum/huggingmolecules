from typing import Optional

from experiments.training import tune_hyper
from experiments.training.training_benchmark_utils import print_benchmark_results, set_default_experiment_name


def benchmark(prefix: Optional[str] = None, results_only: bool = False):
    set_default_experiment_name(prefix)

    if not results_only:
        tune_hyper()

    print_benchmark_results()
