import argparse

import gensim

from config.default import get_cfg_defaults
from runner import FirstRunner

import warnings


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--use-comet', action="store_true", default=False)
    parser.add_argument('--start-from', type=int, default=-1)
    parser.add_argument('--eval-epoch', type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file("config/" + args.config + ".yaml")
    cfg.freeze()
    runner = FirstRunner(cfg, args.use_comet)
    runner.load_model()
    # runner.train()
    runner.eval()
