import argparse

import gensim

from config.default import get_cfg_defaults
from runner import MainRunner

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
    runner = MainRunner(cfg, args.use_comet)
    if args.eval_epoch != -1:
        runner.load_model(cfg.train.saved_path + "/model-{}.pt".format(args.eval_epoch))
        runner.eval(args.eval_epoch, "Test", display_interval=50)
    else:
        if args.start_from >= 0:
            runner.load_model(cfg.train.saved_path + "/model-{}.pt".format(args.start_from))
        runner.train()
