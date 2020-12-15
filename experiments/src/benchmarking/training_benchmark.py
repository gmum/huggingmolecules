from typing import Optional

import gin

from experiments.src import tune_hyper
from experiments.src.benchmarking.training_benchmark_utils import print_benchmark_results, set_default_experiment_name


@gin.configurable('benchmark')
def benchmark(test_metric: str, prefix: Optional[str] = None, results_only: bool = False):
    set_default_experiment_name(prefix)

    if not results_only:
        tune_hyper()

    print_benchmark_results(test_metric=test_metric)
