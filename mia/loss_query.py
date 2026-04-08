from data.utils import load_dataset
from generative_models import AbstractGenerativeModel, VAE
from generative_models.utils import load_model
from . import path_utils
import utils
from utils import index_to_mask

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

class LossQuery:

    def __init__(self, batch_size, device, n_loss_samples):
        self.batch_size = batch_size
        self.device = device
        self.n_loss_samples = n_loss_samples

    def load_model(self, path):
        model, train_indices = load_model(
            path=path,
            device=self.device,
        )
        if isinstance(model, VAE):
            model.n_rsamples = self.n_loss_samples
        return model, train_indices

    def loss_signal(self, audit_loader, model):
        sig = []
        for samples in tqdm(audit_loader, total=len(audit_loader), desc=f"Computing loss signal"):
            samples = samples.to(self.device)
            sig.append(self.compute_averaged_loss(model, samples))
        sig = torch.concat(sig, dim=0)
        assert sig.shape == (len(audit_loader.dataset),)
        return sig

    @torch.inference_mode()
    def compute_averaged_loss(self, model: AbstractGenerativeModel, samples: torch.Tensor):
        '''
        Compute averaged per-sample losses for one batch.
        Args:
            model (AbstractGenerativeModel): Generative model used for querying.
            samples (torch.Tensor): Batch of samples on the evaluation device.
        Returns:
            torch.Tensor: Averaged per-sample losses on CPU.
        '''
        batch_size = samples.shape[0]
        match model.__class__.__name__:
            case "DDPM":
                loss_samples = []
                step_index = int(model.time_steps * 0.1)
                for _ in range(self.n_loss_samples):
                    t = torch.full(size=(batch_size,), fill_value=step_index, device=samples.device, dtype=torch.long)
                    loss = model.per_sample_loss(samples, t).cpu()
                    loss_samples.append(loss)
                avg_loss = torch.stack(loss_samples).mean(dim=0)
            case "FlowMatching":
                loss_samples = []
                query_time = 0.25 # TODO: Find optimal query time
                for _ in range(self.n_loss_samples):
                    t = torch.full(size=(batch_size,), fill_value=query_time, device=samples.device, dtype=torch.float32)
                    loss = model.per_sample_loss(samples, t).cpu()
                    loss_samples.append(loss)
                avg_loss = torch.stack(loss_samples).mean(dim=0)
            case "VAE":
                avg_loss = model.per_sample_loss(samples).cpu()
            case _:
                raise ValueError("Unavailable class of generative model.")
        return avg_loss

    def query_loss(self, dataset, model_path):
        audit_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        model, train_indices = self.load_model(model_path)
        mask = index_to_mask(train_indices, n_indices=len(dataset))
        sig = self.loss_signal(audit_loader, model)
        return sig.to(dtype=torch.float32), mask.to(dtype=torch.bool)

def save_loss_signals(res_dir, target_path, loss_sig, train_mask, n_loss_samples):
    '''
    Save one checkpoint's loss-signal payload.
    Args:
        res_dir (str | Path): Audit results directory.
        target_path (str | Path): Target checkpoint path.
        loss_sig (torch.Tensor): Loss signal values to save.
        train_mask (torch.Tensor): Training-membership mask.
        n_loss_samples (int): Number of loss samples used per point.
    Returns:
        Path: Saved pickle path.
    '''
    target_path = Path(target_path)
    output_dir = path_utils.loss_signals_dir(res_dir, target_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / path_utils.loss_signals_pickle_name(target_path, n_loss_samples)
    with open(output_path, "wb") as file:
        pickle.dump({
            "loss_sigs": loss_sig.tolist(),
            "train_mask": train_mask.tolist(),
        }, file)
    return output_path

def migrate_lira_scores_to_loss_signals(res_dir, score_paths, n_loss_samples):
    '''
    Create loss-signal pickles from existing LiRA score files.
    Args:
        res_dir (str | Path): Audit results directory.
        score_paths (list[str]): LiRA score files or folders.
        n_loss_samples (int): Number of loss samples recorded in the output name.
    Returns:
        list[Path]: Saved loss-signal pickle paths.
    '''
    saved_paths = []
    resolved_config = utils.Config({
        "res_dir": res_dir,
        "score_paths": score_paths,
    })
    for score_path in path_utils.resolve_audit_score_paths(resolved_config):
        with open(score_path, "rb") as file:
            payload = pickle.load(file)
        if "loss_sigs" not in payload or "train_mask" not in payload:
            raise ValueError(f"LiRA score pickle is missing loss signals or train mask: {score_path}")
        loss_sigs = torch.tensor(payload["loss_sigs"], dtype=torch.float32)
        if loss_sigs.ndim == 2:
            loss_sig = loss_sigs[0]
        elif loss_sigs.ndim == 1:
            loss_sig = loss_sigs
        else:
            raise ValueError(f"Unexpected loss_sigs rank in {score_path}: {loss_sigs.ndim}")
        train_mask = torch.tensor(payload["train_mask"], dtype=torch.bool)
        target_path = path_utils.target_checkpoint_path_from_scores_pickle_path(score_path)
        saved_paths.append(save_loss_signals(res_dir, target_path, loss_sig, train_mask, n_loss_samples))
    return saved_paths

def run_loss_query(config, device, checkpoint_paths_override=None, lira_score_paths_override=None):
    '''
    Compute or migrate loss-signal pickles.
    Args:
        config (Config): Loss-query configuration.
        device (torch.device): Device used for model evaluation.
        checkpoint_paths_override (list[str] | None): Optional checkpoint overrides.
        lira_score_paths_override (list[str] | None): Optional LiRA score overrides.
    Returns:
        list[Path]: Saved loss-signal pickle paths.
    '''
    if not hasattr(config, "n_loss_samples"):
        raise ValueError("Config must define n_loss_samples.")
    n_loss_samples = int(config.n_loss_samples)
    saved_paths = []
    checkpoint_paths = checkpoint_paths_override if checkpoint_paths_override is not None else getattr(config, "checkpoint_paths", None)
    if checkpoint_paths:
        image_size = utils.parse_properties_from_checkpoint_path(checkpoint_paths[0])["size"]
        dataset = load_dataset(config.dataset, data_dir=config.data_dir, size=image_size)
        loss_query = LossQuery(
            batch_size=config.batch_size,
            device=device,
            n_loss_samples=n_loss_samples,
        )
        for checkpoint_path in map(Path, checkpoint_paths):
            loss_sig, train_mask = loss_query.query_loss(dataset, checkpoint_path)
            saved_paths.append(save_loss_signals(config.res_dir, checkpoint_path, loss_sig, train_mask, n_loss_samples))

    lira_score_paths = lira_score_paths_override if lira_score_paths_override is not None else getattr(config, "lira_score_paths", None)
    if lira_score_paths:
        saved_paths.extend(migrate_lira_scores_to_loss_signals(config.res_dir, lira_score_paths, n_loss_samples))

    if not saved_paths:
        raise ValueError("No checkpoint_paths or lira_score_paths specified.")
    return saved_paths

def parse_args(argv=None):
    '''
    Parse CLI arguments for loss querying.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Compute and save loss signals for model checkpoints.")
    default_config_path = str(utils.resolve_path(Path("mia") / "configs" / "config_loss_query.yaml", utils.get_root()))
    parser.add_argument(
        "--config",
        default=default_config_path,
        help="Path to loss query config yaml file.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        default=None,
        help="Checkpoint file paths. Overrides config.",
    )
    parser.add_argument(
        "--lira-score-paths",
        nargs="+",
        default=None,
        help="Existing LiRA score pickle files or folders to migrate. Overrides config.",
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
    config_path = utils.resolve_path(args.config, utils.get_root())
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    config = utils.Config(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_loss_query(
        config=config,
        device=device,
        checkpoint_paths_override=args.checkpoint_paths,
        lira_score_paths_override=args.lira_score_paths,
    )

if __name__ == "__main__":
    main()
