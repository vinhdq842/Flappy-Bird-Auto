from argparse import ArgumentParser
from dataclasses import dataclass

import yaml
from simple_parsing.helpers import Serializable


@dataclass
class ModelArgs(Serializable):
    n_actions: int
    n_temp_frames: int
    p_drop: float


@dataclass
class TrainingArgs(Serializable):
    seed: int
    lr: float
    gamma: float
    batch_size: int
    replay_memory_size: int
    init_eps: float
    final_eps: float
    threshold: float
    self_training_ratio: float
    num_steps: int
    checkpoint_interval: int
    replay_interval: int
    log_dir: str
    log_interval: int
    copy_interval: int
    save_dir: str
    checkpoint_name: str


@dataclass
class TestArgs(Serializable):
    best_checkpoint: str


@dataclass
class GameArgs(Serializable):
    bird_type: str
    pipe_type: str
    background_type: str
    show_background: bool
    show_point: bool


@dataclass
class Configs(Serializable):
    device: str
    fps: int
    model: ModelArgs
    training: TrainingArgs
    test: TestArgs
    game: GameArgs


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

    return Configs.from_dict(config)
