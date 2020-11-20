import argparse

import gin


def apply_gin_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-files', type=str, required=True)
    parser.add_argument('-b', '--bindings', type=str, required=False, default=None,
                        help='List of bindings separated by \'#\', eg.: --bindings "name.param1 = 10# name.param2 = \'path\'"')
    args = parser.parse_args()
    configs = args.config_files.split("#")
    bindings = args.bindings.split("#") if args.bindings else None
    gin.parse_config_files_and_bindings(configs, bindings)


def parse_gin_str(gin_str):
    import io
    buf = io.StringIO(gin_str)
    C = {}
    for line in buf.readlines():
        if len(line.strip()) == 0 or line[0] == "#":
            continue

        k, v = line.split("=")
        k, v = k.strip(), v.strip()
        k1, k2 = k.split(".")

        v = eval(v)

        C[k1 + "." + k2] = v
    return C
