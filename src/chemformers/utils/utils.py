import gin
import argparse


def apply_gin_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, required=True)
    parser.add_argument('-b', '--bindings', type=str, required=False, default=None,
                        help='List of bindings separated by \'#\', eg.: --bindings "name.param1 = 10# name.param2 = \'path\'"')
    args = parser.parse_args()
    gin.parse_config_file(args.config_file)
    if args.bindings:
        gin.parse_config(args.bindings.split("#"))
