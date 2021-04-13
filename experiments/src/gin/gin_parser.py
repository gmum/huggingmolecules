import argparse
import os
from typing import Optional

from experiments.src.gin.gin_parser_utils import _parse_gin_config_flags, _append_gin_config_flags, \
    _append_additional_bindings_flags, _parse_additional_bindings_flags

CONFIGS_ROOT = os.path.join('experiments', 'configs')


def eval_or_str(x):
    try:
        return eval(x)
    except NameError:
        return str(x)


ADDITIONAL_BINDINGS_ARGS = {
    'name.prefix': {'type': str},
    'model.pretrained_name': {'type': eval_or_str},
    'train.gpus': {'type': eval},
    'train.num_epochs': {'type': int}
}


def parse_gin_config_files_and_bindings(*,
                                        base: str = None,
                                        dataset: str = None,
                                        model: str = None,
                                        parser: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    if not parser:
        parser = argparse.ArgumentParser()

    _append_gin_config_flags(parser, base, dataset, model)
    _append_additional_bindings_flags(parser, ADDITIONAL_BINDINGS_ARGS)

    args = parser.parse_args()
    _parse_gin_config_flags(args, CONFIGS_ROOT)
    _parse_additional_bindings_flags(args, ADDITIONAL_BINDINGS_ARGS)

    return args
