from typing import Optional

import gin
import functools


def print_benchmark_results(test_metric: str):
    user_name = gin.query_parameter('neptune.user_name')
    project_name = gin.query_parameter('neptune.project_name')
    experiment_name = gin.query_parameter('optuna.study_name')
    hps_dict = gin.query_parameter('optuna.params')
    hps_list = [hps for hps in hps_dict.keys() if hps != 'data.split_seed']

    import neptune
    project = neptune.init(f'{user_name}/{project_name}')
    data = project.get_leaderboard(state='succeeded')
    data = data[data['name'] == experiment_name]

    no_trials = len(data)
    total_no_trials = functools.reduce(lambda a, b: a * b,
                                       map(len, (v.__deepcopy__(None)
                                                 if isinstance(v, gin.config.ConfigurableReference)
                                                 else v for v in hps_dict.values())))

    if total_no_trials <= no_trials:
        print(f'BENCHMARK IS FINISHED ({no_trials}/{total_no_trials})')
    else:
        print(f'BENCHMARK IS RUNNING ({no_trials}/{total_no_trials})')

    valid_metric = f'channel_valid_loss'
    test_metric = f'channel_{test_metric}'

    data[valid_metric] = data[valid_metric].astype(float)
    data[test_metric] = data[test_metric].astype(float)

    grouped = data.groupby([f'parameter_{hps}' for hps in hps_list])
    aggregated = grouped[[valid_metric, test_metric]].agg(['mean', 'std'])
    best_hps = aggregated[valid_metric]['mean'].idxmin()
    result = aggregated.loc[best_hps]

    print(result)
    print('Best hps:')
    best_hps = [best_hps] if not isinstance(best_hps, tuple) else best_hps
    for hps, val in zip(hps_list, best_hps):
        print(f'  {hps} = {val}')

    mean = result[test_metric]["mean"]
    std = result[test_metric]["std"]
    print(f'Result: {round(mean, 3)} \u00B1 {round(std, 3)}')


def set_default_experiment_name(prefix: Optional[str] = None):
    with gin.unlock_config():
        model_name = gin.query_parameter('model.cls_name')
        task_name = gin.query_parameter('data.task_name')
        dataset_name = gin.query_parameter('data.dataset_name')
        prefix = f'{prefix}_' if prefix else ""
        gin.bind_parameter('optuna.study_name', f'{prefix}{model_name}_{task_name}_{dataset_name}')
