import argparse
import os
import shutil
import sys

import yagmail

from_email = 'kowalski.jan.development@gmail.com'
to_email = 'rumcajsgajos@gmail.com'

all_datasets = ['freesolv', 'caco2', 'clearance', 'hia', 'bioavailability', 'ppbr']

mat_best_lr = []
chemberta_best_lr = []
chemprop_best_lr = [1.0E-4, 1.0E-4, 1.0E-6, 5.0E-4, 1.0E-4, 5.0E-4]

targets_dict = {
    'mat_ensemble2': [
        "python -m experiments.benchmark2 --name.prefix Ensemble2 "
        "-m mat "
        f"-d {dataset} "
        "-b train.gpus=[?]#"
        f"optimizer.lr={lr}"
        for dataset, lr in zip(all_datasets, mat_best_lr)
    ],
    'chemberta_ensemble2': [
        "python -m experiments.benchmark2 --name.prefix Ensemble2 "
        "-m chemberta "
        f"-d {dataset} "
        "-b train.gpus=[?]#"
        f"optimizer.lr={lr}"
        for dataset, lr in zip(all_datasets, chemberta_best_lr)
    ],
    'chemprop_ensemble2': [
        "python -m experiments.benchmark2 --name.prefix Ensemble2 "
        "-m chemprop "
        f"-d {dataset} "
        "-b train.gpus=[?]#"
        f"optimizer.lr={lr}"
        for dataset, lr in zip(all_datasets, chemprop_best_lr)
    ],
    'finalize': [
        f'python -m experiments.train --name.prefix "{prefix}" '
        f"-m {model} "
        f"-d {dataset} "
        "-b train.gpus=0#"
        f"optimizer.lr={lr}#"
        f"data.split_seed={seed}#"
        f"dummy.trial={trial}#"
        "neptune.project_name=\\'Benchmarks\\'#"
        f"train.num_workers=0 ; rm -rf experiments_results/{prefix}*"
        for prefix, model, dataset, lr, seed, trial in [
            ('Ensemble2', 'chemprop', 'freesolv', 1.0E-4, 13, 3)
        ]
    ],
    'test_ensemble2': [
        "python -m experiments.benchmark2 --name.prefix Ensemble2 "
        "--benchmark.compute_results --benchmark.ensemble_max_size None --benchmark.ensemble_pick_method all "
        "-m chemprop "
        f"-d {dataset} "
        "-b train.gpus=[3]#"
        f"optimizer.lr={lr}"
        for dataset, lr in zip(all_datasets[:1], chemprop_best_lr[:1])
    ],

    'mat': [
        "python -m experiments.benchmark --results_only --ensemble --pick_method greedy "
        "-m mat "
        f"-d {dataset} "
        "-b train.gpus=[0]"
        for dataset in ['caco2', 'hia', 'bioavailability', 'ppbr']
    ],
    'chemberta_clearance': [
                               "python -m experiments.benchmark "
                               "-m chemberta "
                               f"-d {dataset} "
                               "-b train.gpus=[0]"
                               for dataset in ['hia', 'bioavailability', 'ppbr']
                           ] + [
                               "python -m experiments.benchmark "
                               f"-m {model} "
                               "-d clearance "
                               "-b train.gpus=[0]"
                               for model in ['chemberta', 'mat']
                           ]
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
            os.remove(stderr_file)
            break
        trial_no += 1

    if trial_no > max_trials_num:
        print("\nMaximal number of trials exceeded! Shutting down...")
        if notify:
            send_notification_failed(cmd, stderr_file)
        sys.exit()

if notify:
    send_notification_succeed()
