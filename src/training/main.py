from __future__ import annotations

import os
from argparse import ArgumentParser
from env.SingleStockEnv import SingleStockEnv 
import warnings
# construct environment


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


# "./" will be added in front of each directory
def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)


def main() -> int:
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = build_parser()
    options = parser.parse_args()
    warnings.filterwarnings("ignore")

    if options.mode == "train":
        from traderl.train import train

        env = SingleStockEnv

        # demo for elegantrl
        kwargs = (
            {}
        )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        train(env=env)
    else:
        raise ValueError("Wrong mode.")
    return 0


# Users can input the following command in terminal
# python main.py --mode=train
if __name__ == "__main__":
    raise SystemExit(main())

