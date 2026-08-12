from data.utils import infer_dataset_name, load_dataset
from generative_models import AbstractGenerativeModel, AbstractDiffusionModel, VAE
from generative_models.utils import load_model
from . import path_utils
import utils
from utils import index_to_mask

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

class LossQuery:

    def __init__(self, batch_size, device, n_loss_samples, noise_level=0.1):
        self.batch_size = batch_size
        self.device = device
        self.n_loss_samples = n_loss_samples
        self.noise_level = noise_level

    def load_model(self, path):
        model, train_indices = load_model(
            path=path,
            device=self.device,
        )
        if isinstance(model, VAE):
            model.n_rsamples = 1
        return model, train_indices

    def loss_signal(self, audit_loader, model):
        sigs = []
        for samples in tqdm(audit_loader, total=len(audit_loader), desc=f"Computing loss signal"):
            samples = samples.to(self.device)
            sigs.append(self.compute_loss_samples(model, samples))
        sigs = torch.concat(sigs, dim=0)
        assert sigs.shape == (len(audit_loader.dataset), self.n_loss_samples)
        return sigs

    @torch.inference_mode()
    def compute_loss_samples(self, model: AbstractGenerativeModel, samples: torch.Tensor):
        '''
        Compute sorted per-query loss samples for one batch.
        Args:
            model (AbstractGenerativeModel): Generative model used for querying.
            samples (torch.Tensor): Batch of samples on the evaluation device.
        Returns:
            torch.Tensor: Sorted loss samples on CPU with shape (batch_size, n_loss_samples).
        '''
        loss_samples = []
        for _ in range(self.n_loss_samples):
            match model.__class__.__name__:
                case "DDPM" | "FlowMatching":
                    assert isinstance(model, AbstractDiffusionModel)
                    loss = model.fixed_noise_level_per_sample_loss(samples, self.noise_level).cpu()
                case "VAE":
                    loss = model.per_sample_loss(samples).cpu()
                case _:
                    raise ValueError("Unavailable class of generative model.")
            loss_samples.append(loss)
        loss_samples = torch.stack(loss_samples, dim=1)
        sorted_loss_samples = torch.sort(loss_samples, dim=1).values
        return sorted_loss_samples

    def query_loss(self, dataset, model_path, n_data_points=None):
        audit_dataset = dataset if n_data_points is None else Subset(dataset, range(n_data_points))
        audit_loader = DataLoader(audit_dataset, batch_size=self.batch_size, shuffle=False)
        model, train_indices = self.load_model(model_path)
        mask = index_to_mask(train_indices, n_indices=len(dataset))
        if n_data_points is not None:
            mask = mask[:n_data_points]
        sig = self.loss_signal(audit_loader, model)
        return sig.to(dtype=torch.float32), mask.to(dtype=torch.bool)

def save_loss_signals(res_dir, target_path, loss_sig, train_mask, n_loss_samples, noise_level=0.1, n_data_points=None):
    '''
    Save one checkpoint's loss-signal payload.
    Args:
        res_dir (str | Path): Audit results directory.
        target_path (str | Path): Target checkpoint path.
        loss_sig (torch.Tensor): Loss signal values to save.
        train_mask (torch.Tensor): Training-membership mask.
        n_loss_samples (int): Number of loss samples used per point.
        noise_level (float): Query noise level used for loss signals.
        n_data_points (int | None): Optional number of queried data points.
    Returns:
        Path: Saved pickle path.
    '''
    target_path = Path(target_path)
    output_dir = path_utils.loss_signals_dir(res_dir, target_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = path_utils.loss_signals_pickle_name(target_path, n_loss_samples, noise_level, n_data_points)
    output_path = output_dir / output_name
    with open(output_path, "wb") as file:
        pickle.dump({
            "loss_sigs": loss_sig.tolist(),
            "train_mask": train_mask.tolist(),
        }, file)
    return output_path

def resolve_loss_query_dataset(dataset, checkpoint_properties):
    '''
    Resolve the canonical dataset name used for loss querying.
    Args:
        dataset (str | None): Optional dataset name override.
        checkpoint_properties (dict): Metadata parsed from a checkpoint path.
    Returns:
        str: Canonical dataset name accepted by load_dataset.
    '''
    dataset_name = checkpoint_properties["dataset"] if dataset is None else dataset
    return infer_dataset_name(dataset_name)

def run_loss_query(checkpoint_paths, checkpoint_properties, dataset, data_dir, batch_size, res_dir, n_loss_samples, device, noise_level, n_data_points=None):
    '''
    Compute loss-signal pickles for model checkpoints.
    Args:
        checkpoint_paths (list[Path]): Checkpoint paths to query.
        checkpoint_properties (dict): Metadata parsed from the first checkpoint path.
        dataset (str | None): Optional dataset name. Defaults to the first checkpoint's parsed dataset.
        data_dir (str | Path): Dataset directory.
        batch_size (int): Loss-query batch size.
        res_dir (str | Path): Directory used for saved loss signals.
        n_loss_samples (int): Number of loss samples used per point.
        device (torch.device): Device used for model evaluation.
        noise_level (float): Query noise level used for loss signals.
        n_data_points (int | None): Optional number of leading data points to query.
    Returns:
        list[Path]: Saved loss-signal pickle paths.
    '''
    dataset_name = resolve_loss_query_dataset(dataset, checkpoint_properties)
    image_size = checkpoint_properties["size"]
    loaded_dataset = load_dataset(dataset_name, data_dir=data_dir, size=image_size)
    loss_query = LossQuery(
        batch_size=batch_size,
        device=device,
        n_loss_samples=n_loss_samples,
        noise_level=noise_level,
    )
    saved_paths = []
    for checkpoint_path in checkpoint_paths:
        loss_sig, train_mask = loss_query.query_loss(loaded_dataset, checkpoint_path, n_data_points)
        saved_path = save_loss_signals(res_dir, checkpoint_path, loss_sig, train_mask, n_loss_samples, noise_level, n_data_points)
        saved_paths.append(saved_path)
    return saved_paths

def print_loss_query_settings(device, dataset, data_dir, batch_size, res_dir, n_loss_samples, checkpoint_paths, noise_level, n_data_points):
    '''
    Print effective loss-query settings.
    Args:
        device (torch.device): Device used for model evaluation.
        dataset (str): Dataset name.
        data_dir (str | Path): Dataset directory.
        batch_size (int): Loss-query batch size.
        res_dir (str | Path): Directory used for saved loss signals.
        n_loss_samples (int): Number of loss samples used per point.
        checkpoint_paths (list[str] | None): Effective checkpoint paths.
        noise_level (float): Effective query noise level.
        n_data_points (int | None): Optional number of leading data points to query.
    Returns:
        None
    '''
    print("Loss query settings")
    print(f"device: {device}")
    print(f"dataset: {dataset}")
    print(f"data_dir: {data_dir}")
    print(f"batch_size: {batch_size}")
    print(f"res_dir: {res_dir}")
    print(f"n_loss_samples: {n_loss_samples}")
    print(f"noise_level: {noise_level}")
    print(f"n_data_points: {n_data_points}")
    print("checkpoint_paths:")
    for checkpoint_path in checkpoint_paths:
        print(f"  - {checkpoint_path}")

def parse_args(argv=None):
    '''
    Parse CLI arguments for loss querying.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Compute and save loss signals for model checkpoints.")
    root = utils.get_root()
    default_data_dir = utils.resolve_path("datasets", root)
    default_res_dir = utils.resolve_path(Path("mia") / "results", root)
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        required=True,
        help="Checkpoint file paths.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name. Defaults to the dataset parsed from the first checkpoint path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Dataset directory. Defaults to root/datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Loss-query batch size.",
    )
    parser.add_argument(
        "--res-dir",
        type=Path,
        default=default_res_dir,
        help="Result directory. Defaults to root/mia/results.",
    )
    parser.add_argument(
        "--n-loss-samples",
        type=int,
        required=True,
        help="Number of loss samples per point.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        required=True,
        help="Loss-query noise level in [0.0, 1.0].",
    )
    parser.add_argument(
        "--n-data-points",
        type=int,
        default=None,
        help="Optionally compute loss signals for only the first N data points.",
    )
    return parser.parse_args(argv)

def main(argv=None):
    '''
    Entry point for the loss-query CLI.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        None
    '''
    args = parse_args(argv)
    assert 0.0 <= args.noise_level <= 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.checkpoint_paths:
        raise ValueError("No checkpoint_paths specified.")
    checkpoint_paths = [Path(checkpoint_path) for checkpoint_path in args.checkpoint_paths]
    checkpoint_properties = utils.parse_properties_from_checkpoint_path(checkpoint_paths[0])
    dataset_name = resolve_loss_query_dataset(args.dataset, checkpoint_properties)
    print_loss_query_settings(
        device=device,
        dataset=dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        res_dir=args.res_dir,
        n_loss_samples=args.n_loss_samples,
        checkpoint_paths=checkpoint_paths,
        noise_level=args.noise_level,
        n_data_points=args.n_data_points,
    )
    run_loss_query(
        checkpoint_paths=checkpoint_paths,
        checkpoint_properties=checkpoint_properties,
        dataset=dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        res_dir=args.res_dir,
        n_loss_samples=args.n_loss_samples,
        device=device,
        noise_level=args.noise_level,
        n_data_points=args.n_data_points,
    )

if __name__ == "__main__":
    main()
