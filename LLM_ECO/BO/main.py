# Author: baichen318@gmail.com


import os
import sys
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "algo")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "utils")
)
from utils import get_configs, parse_args
from algo.boom_explorer import boom_explorer


def main(args, configs):
    # apply CLI overrides to configs for BO run
    if args.episodes is not None:
        configs.setdefault("bo", {})
        configs["bo"]["max-bo-steps"] = args.episodes
    if args.design is not None:
        configs.setdefault("rl-bo", {})
        configs["rl-bo"]["design-name"] = args.design
    if args.num_envs is not None:
        configs.setdefault("rl-bo", {})
        configs["rl-bo"]["num-envs"] = args.num_envs
    if args.use_cuda:
        configs.setdefault("dkl-gp", {})
        configs["dkl-gp"]["use-cuda"] = True
        configs["dkl-gp"]["gpu-id"] = args.gpu_id
    boom_explorer(configs)


if __name__ == "__main__":
    args = parse_args()
    configs = get_configs(args.configs)
    main(args, configs)
