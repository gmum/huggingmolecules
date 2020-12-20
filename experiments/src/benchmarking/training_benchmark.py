from typing import Optional, Literal

import gin

from experiments.src import tune_hyper
from experiments.src.benchmarking.training_benchmark_utils import set_default_study_name, \
    print_benchmark_results_ensemble


@gin.configurable('benchmark')
def benchmark(prefix: Optional[str] = None,
              results_only: bool = False,
              ensemble_max_size: Optional[int] = 1,
              ensemble_pick_method: Literal['brute', 'greedy'] = 'brute'):
    set_default_study_name(prefix)

    if not results_only:
        tune_hyper()

    print_benchmark_results_ensemble(ensemble_max_size=ensemble_max_size, ensemble_pick_method=ensemble_pick_method)
