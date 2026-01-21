from data import data
from generative_models import ddpm
from generative_models import vae
from utils import get_dataset_and_model_from_path

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm.auto import tqdm

import argparse
from pathlib import Path

def fid_score(model, dataloader, device, disable_tqdm=True):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid = fid.set_dtype(torch.float64)
    for real_samples in tqdm(dataloader, disable=disable_tqdm, desc="Generating samples for FID"):
        batch_size = real_samples.shape[0]
        real_samples = real_samples.to(device)
        gen_samples = model.sample(batch_size)
        fid.update(real_samples, real=True)
        fid.update(gen_samples, real=False)
    score = fid.compute()
    return score

def inception_score(model, n_samples, batch_size, disable_tqdm=True):
    is_metric = InceptionScore(normalize=True).to(device)
    n_batches, remainder = divmod(n_samples, batch_size)
    for _ in tqdm(range(n_batches), disable=disable_tqdm, desc="Generating samples for inception score"):
        samples = model.sample(batch_size)
        is_metric.update(samples)
    samples = model.sample(remainder)
    is_metric.update(samples)
    mean, std = is_metric.compute()
    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        choices=["fid", "is"],
        required=True,
        help="Which metric to use."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Which model checkpoint to evaluate."
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./datasets",
        help="Where datasets are stored."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50000,
        required=False,
        help="Number of samples to evaluate (limited by len(dataset) in case of FID)."
    )
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    dataset_name, model_type = get_dataset_and_model_from_path(args.checkpoint)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.model == "ddpm":
        model = ddpm.DDPM(
            beta=checkpoint["beta"],
            channel_mult=checkpoint["channel_mult"],
            image_dim=checkpoint["image_dim"],
            base_channels=checkpoint["base_channels"],
            dropout=checkpoint["dropout"],
            resample_with_conv=checkpoint["resample_with_conv"],
        )
        model.load(checkpoint["model_state_dict"])
    elif args.model == "vae":
        model = vae.VAE(
            in_ch=checkpoint["in_ch"],
            in_dim=checkpoint["in_dim"],
            latent_dim=checkpoint["latent_dim"],
        )
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.metric == "is":
        score_mean, score_std = inception_score(model, args.n_samples, args.batch_size)
        print(f"{score_mean} +- {score_std}")
    else: # FID
        dataset = getattr(data, dataset_name)(data_dir=args.data_dir, transform=None)
        n_samples = min(args.n_samples, len(dataset))
        dataset = Subset(dataset, torch.arange(n_samples))
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        score = fid_score(model, data_loader, device)
        print(f"FID: {score}")
