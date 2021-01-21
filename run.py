import argparse
import os
import sys

import yagmail

from targets import targets_dict, from_email, to_email

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
parser.add_argument('-n', '--max_trials_num', type=int, default=3)
parser.add_argument('--notify', action='store_true', default=False)
parser.add_argument('--show_commands_only', action='store_true', default=False)
args = parser.parse_args()

target_name = args.target
max_trials_num = args.max_trials_num
notify = args.notify
show_commands_only = args.show_commands_only


def send_notification(status):
    contents = [
        '<h2>commands:</h2>',
        '\n\n'.join(targets_dict[target_name])
    ]
    yag.send(to_email, f'Target {args.target} {status}', contents)


def send_notification_stderr(last_command, stderr_path, status):
    try:
        with open(stderr_path) as fp:
            stderr = fp.read()
    except FileExistsError:
        stderr = '?'
    contents = [
        '<h2>last command:</h2>',
        last_command,
        '<h2>stderr:</h2>',
        stderr[:-1000]
    ]
    yag.send(to_email, f'Target {args.target} {status}', contents)


print('Commands:')
print('\n'.join(targets_dict[target_name]))
if show_commands_only:
    sys.exit()

if notify:
    yag = yagmail.SMTP(from_email)
    send_notification('STARTED')

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
                send_notification_stderr(cmd, stderr_file, 'ABORTED')
            sys.exit()
        if ret == 0:
            os.system(f'rm -f {stderr_file}')
            break
        trial_no += 1

    if trial_no > max_trials_num:
        print("\nMaximal number of trials exceeded! Shutting down...")
        if os.path.exists(stderr_file):
            os.system(f'cat {stderr_file}')
        if notify:
            send_notification_stderr(cmd, stderr_file, 'FAILED')
        sys.exit()

if notify:
    send_notification('SUCCEEDED')
