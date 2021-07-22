import argparse
import sys
from contextlib import contextmanager
from typing import Callable

from eir import __version__
from eir.predict import main as main_predict
from eir.train import main as main_train


@contextmanager
def patch_sys_args(patched_args) -> None:
    old_args = sys.argv
    try:
        sys.argv = patched_args
        yield
    finally:
        sys.argv = old_args


def get_main_cl_args():
    parser = argparse.ArgumentParser()

    action_choices = ["train", "predict"]

    parser.add_argument(
        "--action",
        type=str,
        choices=action_choices,
        help="What action to perform.",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Get current version and exit.",
    )

    help_flag = False
    if "--help" in sys.argv:
        sys.argv.remove("--help")
        help_flag = True

    cl_args, external_args = parser.parse_known_args()

    if cl_args.version:
        print(__version__)
        sys.exit(0)

    if cl_args.action not in ["train", "predict"]:
        parser.error(
            f"the following arguments are required: --action {set(action_choices)}"
        )

    if help_flag:
        external_args.extend(["--help"])

    return cl_args, external_args


def choose_action(cl_args) -> Callable:
    action_map = get_action_map()
    action_arg = cl_args.action
    action_fun = action_map[action_arg]
    return action_fun


def get_action_map():
    return {"train": main_train, "predict": main_predict}


def main():
    main_cl_args, rest = get_main_cl_args()

    external_cl_args = [f"eir/{main_cl_args.action}.py"]
    external_cl_args.extend(rest)

    action_func = choose_action(cl_args=main_cl_args)
    with patch_sys_args(patched_args=external_cl_args):
        action_func()


if __name__ == "__main__":
    main()
