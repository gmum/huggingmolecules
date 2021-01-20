import argparse
import os
from typing import List, Optional

import gin

CONFIGS_ROOT = os.path.join('experiments', 'configs')

additional_args = {
    'name.prefix': {'type': str},
    'model.pretrained_name': {'type': str},
    'train.gpus': {'type': eval},
    'train.num_epochs': {'type': int},
    'benchmark.ensemble_max_size': {'type': int},
    'benchmark.ensemble_pick_method': {'type': str},
    'benchmark.prefix_list': {'type': str, 'nargs': '+'},
    'benchmark.models_names_list': {'type': str, 'nargs': '+'}
}


class DefaultValue:
    pass


def apply_gin_config(*,
                     base: str = None,
                     dataset: str = None,
                     model: str = None,
                     configs_root: str = CONFIGS_ROOT,
                     parser: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    if not parser:
        parser = argparse.ArgumentParser()

    if not base:
        parser.add_argument('-bs', '--base', type=str, default=None)
    if not dataset:
        parser.add_argument('-d', '--dataset', type=str, default=None)
    if not model:
        parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-c', '--config-files', type=str, default='')
    parser.add_argument('-b', '--bindings', type=str, default='')

    if additional_args:
        for param, kwargs in additional_args.items():
            kwargs.update({'default': DefaultValue()})
            parser.add_argument(f'--{param}', **kwargs)

    args = parser.parse_args()

    base = base if base else args.base
    dataset = dataset if dataset else args.dataset
    model = model if model else args.model

    base_config = os.path.join(configs_root, 'bases', f'{base}.gin') if base else ''
    dataset_config = os.path.join(configs_root, 'datasets', f'{dataset}.gin') if dataset else ''
    model_config = os.path.join(configs_root, 'models', f'{model}.gin') if model else ''

    configs = [base_config, dataset_config, model_config]
    configs.extend(args.config_files.split("#"))
    configs = [c for c in configs if len(c) > 0]

    bindings = args.bindings.split("#")
    gin.parse_config_files_and_bindings(configs, bindings)

    if additional_args:
        for param in additional_args.keys():
            val = getattr(args, param)
            if not isinstance(val, DefaultValue):
                with gin.unlock_config():
                    gin.bind_parameter(param, val)

    return args


def get_formatted_config_str(excluded: Optional[List[str]] = None):
    config_map = gin.config._CONFIG
    if excluded:
        config_map = {k: v for k, v in config_map.items() if all(x not in k[1] for x in excluded)}
    return gin.config._config_str(config_map)


def parse_gin_str(gin_str):
    gin_dict = {}
    parser = gin.config.config_parser.ConfigParser(gin_str, gin.config.ParserDelegate())
    for statement in parser:
        scope, selector, arg_name, value, location = statement
        gin_dict[f"{selector}.{arg_name}"] = value
    return gin_dict


@gin.configurable('dummy')
def dummy_function_just_for_dummy_gin_params(**kwargs):
    pass


@gin.configurable('name')
def get_default_name(prefix: str = "",
                     model_name: Optional[str] = None,
                     task_name: Optional[str] = None,
                     dataset_name: Optional[str] = None,
                     full_name: Optional[str] = None):
    if full_name:
        return full_name
    prefix = f'{prefix}_' if len(prefix) > 0 else ""
    model_name = model_name if model_name else gin.query_parameter('model.cls_name')
    task_name = task_name if task_name else gin.query_parameter('data.task_name')
    dataset_name = dataset_name if dataset_name else gin.query_parameter('data.dataset_name')
    return f'{prefix}{model_name}_{task_name}_{dataset_name}'
