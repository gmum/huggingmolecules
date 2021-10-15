import glob
import os


def remove_matching_wildcards(directory: str, file_name: str):
    for file in glob.glob(os.path.join(directory, file_name)):
        print(f'Removing {file}')
        os.remove(file)


if __name__ == '__main__':
    from .huggingmolecules.downloading.downloading_utils import HUGGINGMOLECULES_CACHE
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', action='store_true', default=False)
    parser.add_argument('--models', action='store_true', default=False)
    parser.add_argument('--encodings', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('No arguments provided!')

    cache_dir = os.path.expanduser(HUGGINGMOLECULES_CACHE)
    if args.configs or args.all:
        remove_matching_wildcards(cache_dir, '*.json')
        remove_matching_wildcards(cache_dir, '*.json.lock')
    if args.models or args.all:
        remove_matching_wildcards(cache_dir, '*.pt')
        remove_matching_wildcards(cache_dir, '*.pt.lock')
    if args.encodings or args.all:
        from experiments.src.training.training_utils import HUGGINGMOLECULES_ENCODINGS_CACHE

        encodings_dir = os.path.expanduser(HUGGINGMOLECULES_ENCODINGS_CACHE)
        remove_matching_wildcards(encodings_dir, '*')
