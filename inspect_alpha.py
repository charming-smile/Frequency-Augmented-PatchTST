"""Print the learned alpha of every PatchTST_FreqEmbed checkpoint."""
import argparse
import glob
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="./checkpoints")
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.checkpoints, "**", "checkpoint.pth"),
                  recursive=True)
    )
    if not paths:
        print("no checkpoints found under", args.checkpoints)
        return

    for path in paths:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            keys = [k for k in state if k == "alpha" or k.endswith(".alpha")]
            if keys:
                alpha = float(state[keys[0]])
                print(path, "alpha=", round(alpha, 4))
            else:
                print(path, "no alpha (not a PatchTST_FreqEmbed checkpoint)")
        else:
            print(path, "no alpha (not a PatchTST_FreqEmbed checkpoint)")


if __name__ == "__main__":
    main()
