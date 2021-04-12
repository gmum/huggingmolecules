from experiments.src.benchmarking.benchmarking_utils import get_results_dict, check_results_dict, compute_results, \
    print_results


def benchmark_1_results():
    results_dict = get_results_dict()
    check_results_dict(results_dict)
    results = compute_results(results_dict)
    print_results(*results)
