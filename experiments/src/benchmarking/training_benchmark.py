from typing import Optional

import gin

from experiments.src import tune_hyper
from experiments.src.benchmarking.training_benchmark_utils import set_default_study_name, \
    print_benchmark_results_ensemble


@gin.configurable('benchmark')
def benchmark(prefix: Optional[str] = None, results_only: bool = False, ensemble: bool = False):
    set_default_study_name(prefix)

    if not results_only:
        tune_hyper()

    print_benchmark_results_ensemble(max_ensemble_size=None if ensemble else 1)
