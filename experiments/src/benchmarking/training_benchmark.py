from typing import Optional, Literal

import gin

from experiments.src import tune_hyper
from experiments.src.benchmarking.training_benchmark_utils import print_benchmark_results_ensemble


@gin.configurable('benchmark')
def benchmark(run_benchmark: bool = True,
              compute_results: bool = False,
              ensemble_max_size: Optional[int] = 1,
              ensemble_pick_method: Literal['brute', 'greedy'] = 'brute'):
    if run_benchmark:
        tune_hyper()

    if compute_results:
        print_benchmark_results_ensemble(ensemble_max_size=ensemble_max_size,
                                         ensemble_pick_method=ensemble_pick_method)
