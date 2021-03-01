from typing import Optional, Literal, List

import gin

from experiments.src.benchmarking.benchmarking_benchmark_1_utils import pick_best_ensemble, print_results
from experiments.src.benchmarking.benchmarking_utils import get_names_list, fetch_data


@gin.configurable('benchmark')
def benchmark_1_results(ensemble_max_size: Optional[int] = 1,
                        ensemble_pick_method: Literal['brute', 'greedy', 'all'] = 'brute',
                        prefix_list: Optional[List[str]] = None,
                        models_names_list: Optional[List[str]] = None):
    names_list = get_names_list(prefix_list, models_names_list)

    models_list, targets = fetch_data(names_list)

    ensemble = pick_best_ensemble(models_list,
                                  targets=targets,
                                  max_size=ensemble_max_size,
                                  method=ensemble_pick_method,
                                  names_list=names_list)

    print_results(ensemble, targets)
