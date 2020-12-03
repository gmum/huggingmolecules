import gin


def print_benchmark_results():
    user_name = gin.query_parameter('neptune.user_name')
    project_name = gin.query_parameter('neptune.project_name')
    experiment_name = gin.query_parameter('optuna.study_name')
    hps_list = [hps for hps in gin.query_parameter('optuna.params').keys() if hps != 'data.split_seed']

    import neptune
    project = neptune.init(f'{user_name}/{project_name}')
    data = project.get_leaderboard(state='succeeded')

    data = data.where(data['name'] == experiment_name)
    data['channel_valid_loss'] = data['channel_valid_loss'].astype(float)
    data['channel_test_loss'] = data['channel_test_loss'].astype(float)
    grouped = data.groupby([f'property_param__{hps}' for hps in hps_list])
    aggregated = grouped[['channel_valid_loss', 'channel_test_loss']].agg(['mean', 'std'])
    best_hps = aggregated['channel_valid_loss']['mean'].idxmin()
    result = aggregated.loc[best_hps]

    print(result)
    print('Best hps:')
    best_hps = [best_hps] if not isinstance(best_hps, tuple) else best_hps
    for hps, val in zip(hps_list, best_hps):
        print(f'  {hps} = {val}')
