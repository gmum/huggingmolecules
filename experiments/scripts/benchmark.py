"""
benchmark.py: this script performs a benchmark based on hyper-parameters tuning (grid-search).
For details see main README.md.
"""

if __name__ == "__main__":
    import argparse
    from experiments.src import tune_hyper
    from experiments.src.benchmarking.benchmarking_benchmark_results import benchmark_result
    from experiments.src.gin import parse_gin_config_files_and_bindings

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_only', action='store_true')

    # Bind values to functions/methods parameters by parsing appropriate gin-config files and bindings.
    args = parse_gin_config_files_and_bindings(base='benchmark', parser=parser)

    if not args.results_only:
        # Perform grid-search with a grid predefined in the experiments/configs/base/benchmark.gin gin-config file.
        tune_hyper()
    else:
        # Compute and print the result based on the previously performed grid-search.
        benchmark_result()
