import argparse
from pathlib import Path

import torch

from utils import load_checkpoint, unwrap_checkpoint_state_dicts


def unwrap_compiled_checkpoint(input_path, output_path=None):
    '''
    Load a checkpoint, unwrap `torch.compile` state-dict keys, and save it.
    Args:
        input_path (str | Path): Input checkpoint path.
        output_path (str | Path | None): Optional output checkpoint path.
    Returns:
        Path: Saved checkpoint path.
    '''
    checkpoint = load_checkpoint(str(input_path), torch.device("cpu"))
    unwrapped_checkpoint = unwrap_checkpoint_state_dicts(checkpoint)
    resolved_output_path = Path(output_path) if output_path is not None else Path(input_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrapped_checkpoint, resolved_output_path)
    return resolved_output_path


def parse_args():
    '''
    Parse CLI arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Unwrap torch.compile ('_orig_mod.') state_dict keys in a checkpoint.")
    parser.add_argument("--input", required=True, help="Input checkpoint path.")
    parser.add_argument("--output", default=None, help="Output checkpoint path. Defaults to --input (overwrite in-place).")
    return parser.parse_args()


def main():
    '''
    Run the checkpoint-unwrapping CLI.
    Returns:
        None
    '''
    args = parse_args()
    output_path = unwrap_compiled_checkpoint(args.input, args.output)
    print(f"Wrote unwrapped checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
