import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import utils


def load_loss_history(checkpoint):
    '''
    Load checkpoint train and validation loss histories.
    Args:
        checkpoint (dict): Checkpoint payload.
    Returns:
        tuple[list[float], list[float]]: Train and validation loss histories.
    '''
    train_losses = checkpoint.get("train_losses")
    val_losses = checkpoint.get("val_losses")
    if train_losses is None or val_losses is None:
        raise ValueError("Checkpoint does not contain both train_losses and val_losses.")
    train_losses = [float(loss) for loss in train_losses]
    val_losses = [float(loss) for loss in val_losses]
    if len(train_losses) != len(val_losses):
        raise ValueError("Checkpoint train_losses and val_losses must have matching lengths.")
    if not train_losses:
        raise ValueError("Checkpoint loss history is empty.")
    return train_losses, val_losses


def plot_loss_curves(checkpoint_path, output_path=None):
    '''
    Plot checkpoint train and validation loss curves.
    Args:
        checkpoint_path (str | Path): Checkpoint file to read.
        output_path (str | Path | None): Optional path for the saved figure.
    Returns:
        Path: Path to the saved figure.
    '''
    checkpoint_path = Path(checkpoint_path)
    checkpoint = utils.load_checkpoint(str(checkpoint_path), torch.device("cpu"))
    train_losses, val_losses = load_loss_history(checkpoint)
    epochs = range(1, len(train_losses) + 1)
    if output_path is None:
        output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_loss_curves.png")
    output_path = Path(output_path)
    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(checkpoint_path.stem)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss curves to {output_path}")
    return output_path


def main(argv=None):
    '''
    Parse CLI args for checkpoint loss plotting.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        None
    '''
    parser = argparse.ArgumentParser(description="Plot train and validation losses stored in a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to load.")
    parser.add_argument("--output", default=None, help="Optional output path for the plot.")
    args = parser.parse_args(argv)
    plot_loss_curves(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
