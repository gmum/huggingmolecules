"""
train.py: this script performs training with the pytorch lightning package.
For details see main README.md.
"""

if __name__ == "__main__":
    from experiments.src import train_model
    from experiments.src.gin import parse_gin_config_files

    # Bind values to functions/methods parameters by parsing appropriate gin-config files.
    parse_gin_config_files(base='train')

    train_model()
