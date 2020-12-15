import argparse
import os
from typing import List, Optional

import gin


def apply_gin_config(parser: argparse.ArgumentParser = None,
                     base: str = None,
                     model: str = None,
                     dataset: str = None,
                     configs_root: str = os.path.join('experiments', 'configs')) -> argparse.Namespace:
    if not parser:
        parser = argparse.ArgumentParser()

    if not base:
        parser.add_argument('-bs', '--base', type=str, default=None)
    if not model:
        parser.add_argument('-m', '--model', type=str, default=None)
    if not dataset:
        parser.add_argument('-d', '--dataset', type=str, default=None)
    parser.add_argument('-c', '--config-files', type=str, default='')
    parser.add_argument('-b', '--bindings', type=str, default='')

    args = parser.parse_args()

    base = base if base else args.base
    model = model if model else args.model
    dataset = dataset if dataset else args.dataset

    base_config = os.path.join(configs_root, 'bases', f'{base}.gin') if base else ''
    model_config = os.path.join(configs_root, 'models', f'{model}.gin') if model else ''
    dataset_config = os.path.join(configs_root, 'datasets', f'{dataset}.gin') if dataset else ''

    configs = [base_config, model_config, dataset_config]
    configs.extend(args.config_files.split(";"))
    configs = [c for c in configs if len(c) > 0]

    bindings = args.bindings.split(";")
    gin.parse_config_files_and_bindings(configs, bindings)
    return args


def get_formatted_config_str(excluded: Optional[List[str]] = None):
    config_map = gin.config._CONFIG
    if excluded:
        config_map = {k: v for k, v in config_map.items() if all(x not in k[1] for x in excluded)}
    return gin.config._config_str(config_map)


def parse_gin_str(gin_str):
    import io
    buf = io.StringIO(gin_str)
    C = {}
    for line in buf.readlines():
        if len(line.strip()) == 0 or line[0] == "#":
            continue

        k, v = line.split("=")
        k, v = k.strip(), v.strip()
        k1, k2 = k.split(".")

        v = eval(v)

        C[k1 + "." + k2] = v
    return C
