import argparse
import itertools
import os
import sys

import yagmail

from_email = 'kowalski.jan.development@gmail.com'
to_email = 'rumcajsgajos@gmail.com'

all_datasets = ['freesolv', 'caco2', 'clearance', 'hia', 'bioavailability', 'ppbr']
all_models = ['mat', 'chemberta', 'chemprop']

mat_best_lr = [5.0E-5, 5.0E-4, 1.0E-6, 1.0E-5, 5.0E-6, 1.0E-5]
chemberta_best_lr = [5.0E-4, 0.001, 0.001, 1.0E-5, 5.0E-6, 5.0E-6]
chemprop_best_lr = [1.0E-4, 1.0E-4, 1.0E-6, 5.0E-4, 1.0E-4, 5.0E-4]

targets_dict = {
    'ensemble2_single': [
        "python -m experiments.benchmark2 --results_only --name.prefix Ensemble2 "
        "--benchmark.ensemble_pick_method all "
        f"--benchmark.ensemble_max_size {size} "
        f"-m {model} "
        f"-d {dataset} "
        for size in [1, 3, 7] for dataset in all_datasets for model in all_models
    ],
    'ensemble2_pairs': [
        "python -m experiments.benchmark2 --results_only --name.prefix Ensemble2 "
        "--benchmark.ensemble_pick_method all "
        f"--benchmark.ensemble_max_size {size} "
        f"--benchmark.models_names_list {' '.join(models)} "
        f"-d {dataset} "
        for size in [6, 14] for dataset in all_datasets for models in itertools.combinations(all_models, 2)
    ],
    'finalize': [
        f'python -m experiments.train --name.prefix "{prefix}" '
        f"-m {model} "
        f"-d {dataset} "
        "-b train.gpus=[0]#"
        f"optimizer.lr={lr}#"
        f"data.split_seed={seed}#"
        f"dummy.trial={trial}#"
        "neptune.project_name=\\'Benchmarks\\'#"
        f" ; rm -rf experiments_results/{prefix}*"
        for prefix, model, dataset, lr, seed, trial in [
            ('', 'mat', 'clearance', 1.0E-4, 413, 0),
            ('', 'mat', 'clearance', 5.0E-5, 454, 0)
        ]
    ],

}

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
parser.add_argument('-n', '--max_trials_num', type=int, default=3)
parser.add_argument('--notify', action='store_true', default=False)
parser.add_argument('--show_commands_only', action='store_true', default=False)
args = parser.parse_args()

target_name = args.target
max_trials_num = args.max_trials_num
notify = args.notify
show_command_only = args.show_commands_only


def send_notification_start():
    contents = [
        '<h2>commands:</h2>',
        '\n\n'.join(targets_dict[target_name])
    ]
    yag.send(to_email, f'Target {args.target} STARTED', contents)


def send_notification_aborted(last_command):
    contents = [
        '<h2>last command:</h2>',
        last_command
    ]
    yag.send(to_email, f'Target {args.target} ABORTED', contents)


def send_notification_failed(last_command, stderr_path):
    with open(stderr_path) as fp:
        stderr = fp.read()
    contents = [
        '<h2>last command:</h2>',
        last_command,
        '<h2>stderr:</h2>',
        stderr
    ]
    yag.send(to_email, f'Target {args.target} FAILED', contents)


def send_notification_succeed():
    contents = [
        '<h2>commands:</h2>',
        '\n\n'.join(targets_dict[target_name])
    ]
    yag.send(to_email, f'Target {args.target} SUCCEED', contents)


if show_command_only:
    print('Commands:')
    print('\n'.join(targets_dict[target_name]))
    sys.exit()

if notify:
    yag = yagmail.SMTP(from_email)
    send_notification_start()

stderr_dir = os.path.join('stderr', target_name)
os.makedirs(stderr_dir, exist_ok=True)
for cmd_idx, cmd in enumerate(targets_dict[target_name], 1):
    trial_no = 1
    while trial_no <= max_trials_num:
        print(f'Running command {cmd_idx} (trial {trial_no}): {cmd}')
        stderr_file = os.path.join(stderr_dir, f'cmd_{cmd_idx}_trial_{trial_no}.txt')
        ret = os.system(f"{cmd} 2> {stderr_file}")

        if ret == 2:
            print("\nAbortion. Shutting down...")
            if notify:
                send_notification_aborted(cmd)
            sys.exit()
        if ret == 0:
            os.system(f'rm -f {stderr_file}')
            break
        trial_no += 1

    if trial_no > max_trials_num:
        print("\nMaximal number of trials exceeded! Shutting down...")
        if notify:
            send_notification_failed(cmd, stderr_file)
        sys.exit()

if notify:
    send_notification_succeed()
