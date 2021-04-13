from experiments.src.benchmarking.benchmarking_utils import get_grid_results_dict, check_grid_results_dict, \
    compute_result, \
    print_result


def benchmark_result():
    grid_results_dict = get_grid_results_dict()
    check_grid_results_dict(grid_results_dict)
    result = compute_result(grid_results_dict)
    print_result(*result)
