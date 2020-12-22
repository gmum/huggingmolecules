from experiments.src.benchmarking import benchmark
from experiments.src.gin import apply_gin_config

additional_args = {
    'name.prefix': {'type': str, 'default': ''},
    'benchmark.run_benchmark': {'action': 'store_true'},
    'benchmark.compute_results': {'action': 'store_true'},
    'benchmark.ensemble_max_size': {'type': eval, 'default': 1},
    'benchmark.ensemble_pick_method': {'type': str, 'default': 'brute'}
}

apply_gin_config(base='benchmark2', additional_args=additional_args)

benchmark()