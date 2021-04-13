from typing import Optional, List, Dict, Any

import gin


@gin.configurable('name')
def get_default_experiment_name(prefix: str = "",
                                model_name: Optional[str] = None,
                                task_name: Optional[str] = None,
                                dataset_name: Optional[str] = None,
                                assay_name: Optional[str] = None,
                                full_name: Optional[str] = None) -> str:
    if full_name:
        return full_name
    prefix = f'{prefix}_' if len(prefix) > 0 else ""
    model_name = model_name if model_name else gin.query_parameter('model.cls_name')
    task_name = task_name if task_name else gin.query_parameter('data.task_name')
    dataset_name = dataset_name if dataset_name else gin.query_parameter('data.dataset_name')
    try:
        assay_name = assay_name if assay_name else gin.query_parameter('data.assay_name')
        dataset_name = f'{dataset_name}_{assay_name}'
    except ValueError:
        pass
    return f'{prefix}{model_name}_{task_name}_{dataset_name}'


def get_formatted_config_str(excluded: Optional[List[str]] = None) -> str:
    config_map = gin.config._CONFIG
    if excluded:
        config_map = {k: v for k, v in config_map.items() if all(x not in k[1] for x in excluded)}
    return gin.config._config_str(config_map)


def parse_gin_str(gin_str: str) -> Dict[str, Any]:
    gin_dict = {}
    parser = gin.config.config_parser.ConfigParser(gin_str, gin.config.ParserDelegate())
    for statement in parser:
        scope, selector, arg_name, value, location = statement
        gin_dict[f"{selector}.{arg_name}"] = value
    return gin_dict


def bind_parameters_from_dict(values_dict: Dict[str, Any]) -> None:
    with gin.unlock_config():
        for param, value in values_dict.items():
            if not param.startswith('ignore.'):
                gin.bind_parameter(param, value)
