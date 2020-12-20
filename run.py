import argparse
import os
import shutil
import sys

import yagmail

from_email = 'kowalski.jan.development@gmail.com'
to_email = 'rumcajsgajos@gmail.com'

all_datasets = ['freesolv', 'caco2', 'clearance', 'hia', 'bioavailability', 'ppbr']

targets_dict = {
    'mat': [
        "python -m experiments.benchmark "
        "-m mat "
        f"-d {dataset} "
        "-b train.gpus=[0]"
        for dataset in all_datasets
    ],
    'chemberta': [
        "python -m experiments.benchmark "
        "-m chemberta "
        f"-d {dataset} "
        "-b train.gpus=[?]"
        for dataset in all_datasets
    ],
    'chemprop': [
        "python -m experiments.benchmark "
        "-m chemprop "
        f"-d {dataset} "
        "-b train.gpus=[3]"
        for dataset in all_datasets
    ],
    'chemprop_50': [
        "python -m experiments.benchmark --prefix 50 "
        "-m chemprop "
        f"-d {dataset} "
        "-b train.gpus=[0]#train.num_epochs=50"
        for dataset in all_datasets
    ],
    'chemprop_r2d': [
        "python -m experiments.benchmark "
        "-m chemprop --prefix R2D_Norm "
        f"-d {dataset} "
        "-b train.gpus=[0]#ChempropFeaturizer.features_generator=[\\'rdkit_2d_normalized\\']"
        for dataset in all_datasets
    ]
}

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
parser.add_argument('-n', '--max_trials_num', type=int, default=3)
parser.add_argument('--notify', action='store_true', default=False)
args = parser.parse_args()

target_name = args.target
max_trials_num = args.max_trials_num
notify = args.notify


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
