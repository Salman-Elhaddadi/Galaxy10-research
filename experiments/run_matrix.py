"""Print the fixed research matrix; --execute explicitly starts full training."""
import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-exploratory-b", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--protocol-manifest", default="")
    args = parser.parse_args(argv)
    script = Path(__file__).resolve().with_name("train.py")
    configs = [("a", condition, seed) for condition in ("clean", "aug") for seed in (0, 1, 2)]
    if args.include_exploratory_b:
        configs += [("b", "clean", seed) for seed in (0, 1, 2)]
    for model, condition, seed in configs:
        command = [sys.executable, str(script), "--model", model, "--condition", condition, "--seed", str(seed), "--device", args.device, "--num-workers", str(args.num_workers)]
        if args.protocol_manifest:
            command += ["--protocol-manifest", args.protocol_manifest]
        print(("EXPLORATORY (over cap): " if model == "b" else "Model A: ") + subprocess.list2cmdline(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
