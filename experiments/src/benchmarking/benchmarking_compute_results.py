from typing import Optional, List, Literal, Union

import gin

from experiments.src.benchmarking.benchmarking_utils import print_results, check_models_list, \
    fetch_data, pick_best_ensemble, get_names_list


@gin.configurable('benchmark')
def compute_benchmark_results(ensemble_max_size: Optional[Union[int, List[int]]] = None,
                              ensemble_pick_method: Literal['brute', 'greedy', 'all'] = 'brute',
                              prefix_list: Optional[List[str]] = None,
                              models_names_list: Optional[List[str]] = None,
                              cache_dir: str = 'benchmark_results'):
    names_list = get_names_list(prefix_list, models_names_list)

    models_list, targets = fetch_data(names_list, cache_dir)

    ensemble = pick_best_ensemble(models_list,
                                  targets=targets,
                                  max_size=ensemble_max_size,
                                  method=ensemble_pick_method,
                                  names_list=names_list)

    print_results(ensemble, targets)
