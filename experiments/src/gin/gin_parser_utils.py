import argparse
import os
from typing import Dict, Any

import gin


# gin config flags

def _append_gin_config_flags(parser: argparse.ArgumentParser,
                             base: str,
                             dataset: str,
                             model: str) -> None:
    parser.add_argument('-bs', '--base', type=str, default=base, required=base is None)
    parser.add_argument('-d', '--dataset', type=str, default=dataset, required=dataset is None)
    parser.add_argument('-m', '--model', type=str, default=model, required=model is None)
    parser.add_argument('-b', '--bindings', type=str, default='')


def _parse_gin_config_flags(args: argparse.Namespace, configs_root: str) -> None:
    base_config_path = os.path.join(configs_root, 'bases', f'{args.base}.gin')
    dataset_config_path = os.path.join(configs_root, 'datasets', f'{args.dataset}.gin')
    model_config_path = os.path.join(configs_root, 'models', f'{args.model}.gin')

    configs = [base_config_path, dataset_config_path, model_config_path]
    configs = [config_path for config_path in configs if len(config_path) > 0]
    bindings = args.bindings.split("#")
    gin.parse_config_files_and_bindings(configs, bindings)


# additional bindings flags

class __DefaultValue:
    pass


def _append_additional_bindings_flags(parser: argparse.ArgumentParser,
                                      additional_bindings_args: Dict[str, Dict[str, Any]]) -> None:
    for param, kwargs in additional_bindings_args.items():
        kwargs.update({'default': __DefaultValue()})
        parser.add_argument(f'--{param}', **kwargs)


def _parse_additional_bindings_flags(args: argparse.Namespace,
                                     additional_binding_args: Dict[str, Dict[str, Any]]) -> None:
    for param in additional_binding_args.keys():
        val = getattr(args, param)
        if not isinstance(val, __DefaultValue):
            with gin.unlock_config():
                gin.bind_parameter(param, val)
