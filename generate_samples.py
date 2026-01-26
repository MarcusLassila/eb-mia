import argparse
import math
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generative_models.utils import load_model


def _prepare_grid_dims(n_images, n_rows=None, n_cols=None):
    ''' Resolve grid dimensions for plotting.

    Args:
        n_images: Number of images to place.
        n_rows: Optional number of rows.
        n_cols: Optional number of columns.

    Returns:
        Tuple[int, int]: Grid rows and columns.
    '''
    if n_rows is None and n_cols is None:
        n_cols = int(math.ceil(math.sqrt(n_images)))
        n_rows = int(math.ceil(n_images / n_cols))
    elif n_rows is None:
        n_rows = int(math.ceil(n_images / n_cols))
    elif n_cols is None:
        n_cols = int(math.ceil(n_images / n_rows))
    if n_rows * n_cols < n_images:
        raise ValueError("Grid is too small for the number of images.")
    return n_rows, n_cols


def _plot_single_image(ax, image):
    ''' Plot a single image on a matplotlib axis.

    Args:
        ax: Matplotlib axis.
        image: Tensor image in CxHxW format.

    Returns:
        None
    '''
    if image.shape[0] == 1:
        ax.imshow(image.squeeze(0), cmap="gray")
    else:
        ax.imshow(image.permute(1, 2, 0))
    ax.axis("off")


def sample_model(model, num_samples, disable_tqdm=True):
    ''' Sample images from a generative model.

    Args:
        model: Generative model instance.
        num_samples: Number of samples to generate.
        disable_tqdm: Whether to disable progress bars when supported.

    Returns:
        torch.Tensor: Generated samples.
    '''
    try:
        return model.sample(num_samples, disable_tqdm=disable_tqdm)
    except TypeError:
        return model.sample(num_samples)


def set_seed(seed):
    ''' Set torch random seeds.

    Args:
        seed: Integer seed value.

    Returns:
        None
    '''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_samples_from_checkpoint(checkpoint_path, num_samples, device, seed=None):
    ''' Load a checkpoint and generate samples.

    Args:
        checkpoint_path: Path to checkpoint.
        num_samples: Number of samples to generate.
        device: Torch device string or object.
        seed: Optional random seed.

    Returns:
        torch.Tensor: Generated samples on CPU.
    '''
    model, _ = load_model(checkpoint_path, torch.device(device))
    if seed is not None:
        set_seed(seed)
    samples = sample_model(model, num_samples)
    return samples.detach().cpu()


def save_samples_raw(samples, output_path):
    ''' Save samples as raw tensor data.

    Args:
        samples: Tensor of samples.
        output_path: Output .pth path.

    Returns:
        Path: Saved path.
    '''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples.detach().cpu(), output_path)
    return output_path


def save_samples_png(samples, output_dir, base_name="samples", grid=False, grid_rows=None, grid_cols=None):
    ''' Save samples as PNG files.

    Args:
        samples: Tensor of samples in NxCxHxW format.
        output_dir: Output directory.
        base_name: Base filename for saved images.
        grid: Whether to save a single grid image.
        grid_rows: Optional grid rows when grid is True.
        grid_cols: Optional grid columns when grid is True.

    Returns:
        List[Path]: List of saved paths.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = samples.detach().cpu().clamp(0.0, 1.0)

    if grid:
        n_rows, n_cols = _prepare_grid_dims(samples.shape[0], grid_rows, grid_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes if hasattr(axes, "flatten") else [axes]
        axes = axes.flatten()
        for idx, ax in enumerate(axes):
            if idx < samples.shape[0]:
                _plot_single_image(ax, samples[idx])
            else:
                ax.axis("off")
        fig.tight_layout()
        output_path = output_dir / f"{base_name}_grid.png"
        fig.savefig(output_path)
        plt.close(fig)
        return [output_path]

    paths = []
    for idx, image in enumerate(samples):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        _plot_single_image(ax, image)
        output_path = output_dir / f"{base_name}_{idx:04d}.png"
        fig.savefig(output_path)
        plt.close(fig)
        paths.append(output_path)
    return paths


def sample_and_save(
    checkpoint_path,
    num_samples,
    output_dir,
    output_format="png",
    grid=False,
    grid_rows=None,
    grid_cols=None,
    base_name="samples",
    device="cpu",
    seed=None,
):
    ''' Sample from a checkpoint and save outputs.

    Args:
        checkpoint_path: Path to checkpoint.
        num_samples: Number of samples to generate.
        output_dir: Output directory.
        output_format: "png" or "raw".
        grid: Whether to save a single grid PNG.
        grid_rows: Optional grid rows when grid is True.
        grid_cols: Optional grid columns when grid is True.
        base_name: Base filename for outputs.
        device: Torch device string.
        seed: Optional random seed.

    Returns:
        List[Path]: Saved file paths.
    '''
    samples = generate_samples_from_checkpoint(checkpoint_path, num_samples, device, seed=seed)
    if output_format == "png":
        return save_samples_png(
            samples,
            output_dir,
            base_name=base_name,
            grid=grid,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
    if output_format == "raw":
        output_path = Path(output_dir) / f"{base_name}.pth"
        return [save_samples_raw(samples, output_path)]
    raise ValueError(f"Unsupported output format: {output_format}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a training checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file.")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--output-format", choices=["png", "raw"], default="png")
    parser.add_argument("--output-dir", type=str, default="./images")
    parser.add_argument("--grid", action="store_true", help="Save a single grid PNG.")
    parser.add_argument("--grid-rows", type=int, default=None)
    parser.add_argument("--grid-cols", type=int, default=None)
    parser.add_argument("--base-name", type=str, default="samples")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = _parse_args()
    sample_and_save(
        args.checkpoint,
        args.num_samples,
        args.output_dir,
        output_format=args.output_format,
        grid=args.grid,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        base_name=args.base_name,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
