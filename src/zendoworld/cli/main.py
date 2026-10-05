"""Dispatch user-facing commands while keeping model dependencies lazy."""
import argparse
import importlib
import sys

COMMANDS = {
    "run": "zendoworld.experiments.runner",
    "compute-log": "zendoworld.experiments.compute_log",
    "compute-eig": "zendoworld.analysis.compute_eig",
    "create-predictions": "zendoworld.synthesis.predictions.create",
}


def main():
    parser = argparse.ArgumentParser(description="ZendoWorld experiments and analysis")
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"zendoworld {args.command}", *sys.argv[2:]]
    importlib.import_module(COMMANDS[args.command]).main()
