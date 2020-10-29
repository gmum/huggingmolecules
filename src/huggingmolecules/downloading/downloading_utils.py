import os

import filelock
import gdown

default_cache_dir = '~/.cache/torch/huggingmolecules/'
HUGGINGMOLECULES_CACHE = os.getenv("HUGGINGMOLECULES_CACHE", default_cache_dir)


def get_cache_filepath(pretrained_name: str, archive_dict: dict, extension: str) -> str:
    file_name = f'{pretrained_name}.{extension}'
    file_path = os.path.join(HUGGINGMOLECULES_CACHE, file_name)
    return os.path.expanduser(file_path)


def download_file(src: str, target: str) -> None:
    dirname = os.path.dirname(target)
    os.makedirs(dirname, exist_ok=True)
    lock_path = target + ".lock"
    with filelock.FileLock(lock_path):
        if not os.path.exists(target):
            gdown.download(src, target)


def from_cache(pretrained_name: str, archive_dict: dict, extension: str) -> str:
    if pretrained_name not in archive_dict:
        return None
    file_path = get_cache_filepath(pretrained_name, archive_dict, extension)
    if not os.path.exists(file_path):
        src = archive_dict[pretrained_name]
        download_file(src, file_path)
    return file_path
