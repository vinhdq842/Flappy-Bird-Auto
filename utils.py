from argparse import ArgumentParser

import yaml


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = Config(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __contains__(self, k):
        return k in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_configs():
    parser = ArgumentParser("Flappy Bird Auto")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/config.yaml",
        help="path to *.yaml config file",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    return Config(**config)
