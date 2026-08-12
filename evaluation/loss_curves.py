import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import utils


def _to_float_list(values):
    return [*map(float(values))]

def load_loss_history(checkpoint):
    '''
    Load checkpoint loss histories.
    Args:
        checkpoint (dict): Checkpoint payload.
    Returns:
        tuple[list[float], list[float], list[float], list[float], list[float]]: Available loss histories.
    '''
    train_losses = _to_float_list(checkpoint.get("train_losses", []))
    train_eval_losses = _to_float_list(checkpoint.get("train_eval_losses", []))
    val_losses = _to_float_list(checkpoint.get("val_losses", []))
    fixed_noise_train_losses = _to_float_list(checkpoint.get("fixed_noise_train_losses", []))
    fixed_noise_val_losses = _to_float_list(checkpoint.get("fixed_noise_val_losses", []))
    assert train_losses
    assert val_losses
    return train_losses, train_eval_losses, val_losses, fixed_noise_train_losses, fixed_noise_val_losses

def _plot_loss_pair(epochs, left_losses, right_losses, left_label, right_label, title, output_path):
    '''
    Save one two-curve loss plot.
    Args:
        epochs (range): Epoch numbers for the loss histories.
        left_losses (list[float]): First loss history.
        right_losses (list[float]): Second loss history.
        left_label (str): Label for the first loss history.
        right_label (str): Label for the second loss history.
        title (str): Plot title.
        output_path (Path): Destination image path.
    Returns:
        Path: Saved image path.
    '''
    plt.figure()
    plt.plot(epochs, left_losses, label=left_label)
    plt.plot(epochs, right_losses, label=right_label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss curves to {output_path}")
    return output_path

def plot_loss_curves(checkpoint_path, output_dir=None):
    '''
    Plot checkpoint loss curves.
    Args:
        checkpoint_path (str | Path): Checkpoint file to read.
        output_dir (str | Path | None): Optional directory for saved figures.
    Returns:
        list[Path]: Paths to saved figures.
    '''
    checkpoint_path = Path(checkpoint_path)
    checkpoint = utils.load_checkpoint(str(checkpoint_path), torch.device("cpu"))
    train_losses, train_eval_losses, val_losses, fixed_noise_train_losses, fixed_noise_val_losses = load_loss_history(checkpoint)
    assert len(train_losses) == len(val_losses)
    output_dir = Path("evaluation") / "loss_curves" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    train_epochs = range(1, len(train_losses) + 1)
    saved_paths.append(_plot_loss_pair(
        train_epochs,
        train_losses,
        val_losses,
        "train",
        "validation",
        f"{checkpoint_path.stem}: train vs validation",
        output_dir / f"{checkpoint_path.stem}_train_vs_val_loss.png",
    ))
    if train_eval_losses:
        assert len(train_eval_losses) == len(val_losses)
        eval_epochs = range(1, len(train_eval_losses) + 1)
        saved_paths.append(_plot_loss_pair(
            eval_epochs,
            train_eval_losses,
            val_losses,
            "train eval",
            "validation",
            f"{checkpoint_path.stem}: train eval vs validation",
            output_dir / f"{checkpoint_path.stem}_train_eval_vs_val_loss.png",
        ))
    if fixed_noise_train_losses or fixed_noise_val_losses:
        assert len(fixed_noise_train_losses) == len(fixed_noise_val_losses)
        fixed_noise_epochs = range(1, len(fixed_noise_train_losses) + 1)
        saved_paths.append(_plot_loss_pair(
            fixed_noise_epochs,
            fixed_noise_train_losses,
            fixed_noise_val_losses,
            "fixed noise train",
            "fixed noise validation",
            f"{checkpoint_path.stem}: fixed noise train vs validation",
            output_dir / f"{checkpoint_path.stem}_fixed_noise_train_vs_val_loss.png",
        ))
    return saved_paths

def main(argv=None):
    '''
    Parse CLI args for checkpoint loss plotting.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        None
    '''
    parser = argparse.ArgumentParser(description="Plot loss curves stored in a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to load.")
    parser.add_argument("--output", default=None, help="Optional output directory for the plots.")
    args = parser.parse_args(argv)
    plot_loss_curves(args.checkpoint, args.output)

if __name__ == "__main__":
    main()
