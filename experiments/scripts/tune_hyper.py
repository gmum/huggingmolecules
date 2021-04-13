"""
tune_hyper.py: this script performs hyper-parameters tuning with the optuna package.
For details see main README.md.
"""

if __name__ == "__main__":
    from experiments.src import tune_hyper
    from experiments.src.gin import parse_gin_config_files_and_bindings

    # Bind values to functions/methods parameters by parsing appropriate gin-config files and bindings.
    parse_gin_config_files_and_bindings(base='tune_hyper')

    tune_hyper()
