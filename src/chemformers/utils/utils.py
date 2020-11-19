import gin
import argparse


def apply_gin_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-files', type=str, required=True)
    parser.add_argument('-b', '--bindings', type=str, required=False, default=None,
                        help='List of bindings separated by \'#\', eg.: --bindings "name.param1 = 10# name.param2 = \'path\'"')
    args = parser.parse_args()
    configs = args.config_files.split("#")
    bindings = args.bindings.split("#") if args.bindings else None
    gin.parse_config_files_and_bindings(configs, bindings)
