"""Convenience runner: executes the known-truth simulation then real-data exploration."""
import argparse
from .simulation import run_simulation
from . import real_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cookie_cats.csv")
    args = parser.parse_args()
    run_simulation()
    import sys
    sys.argv = ["real_data", "--csv", args.csv]
    real_data.main()

if __name__ == "__main__":
    main()
