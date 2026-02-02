import argparse

from . import evaluation
import utils


def default_resdir():
    '''Return default results directory. Args: None. Returns: str.'''
    root = utils.get_root()
    if root is None:
        return "mia/results"
    return f"{root}/mia/results"


def parse_args(argv=None):
    '''Parse CLI arguments for result parsing. Args: argv (list[str]|None). Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Parse MIA results.")
    parser.add_argument(
        "--resdir",
        default=default_resdir(),
        help="Results directory.",
    )
    parser.add_argument(
        "--low-exponent",
        type=int,
        default=-4,
        help="Low exponent for logspace FPR.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    '''Entry point for results parsing CLI. Args: argv (list[str]|None). Returns: None.'''
    args = parse_args(argv)
    evaluation.plot_average_roc_curves(args.resdir, low_exponent=args.low_exponent)


if __name__ == "__main__":
    main()
