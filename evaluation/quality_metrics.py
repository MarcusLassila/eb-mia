from data.utils import load_dataset
from generative_models.utils import load_model
from utils import get_dataset_and_model_from_path

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm.auto import tqdm

import argparse

def _normalize_data(samples):
    return (samples + 1.0) * 0.5

def _validate_sample_assumptions(samples, device, name, tolerance=1e-8):
    assert torch.isfinite(samples).all().item(), f"{name} contains non-finite values"
    min_value = float(samples.min())
    max_value = float(samples.max())
    assert min_value >= -tolerance and max_value <= 1.0 + tolerance, (
        f"{name} out of range [0,1]: min={min_value}, max={max_value}"
    )

def fid_score(model, dataloader, device, disable_tqdm=True):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid = fid.set_dtype(torch.float64)
    checked = False
    for real_samples in tqdm(dataloader, desc="Generating samples for FID"):
        batch_size = real_samples.shape[0]
        real_samples = _normalize_data(real_samples).to(device)
        gen_samples = model.sample(batch_size, disable_tqdm=disable_tqdm)
        if not checked:
            _validate_sample_assumptions(real_samples, device, "real samples")
            _validate_sample_assumptions(gen_samples, device, "generated samples")
            checked = True
        fid.update(real_samples, real=True)
        fid.update(gen_samples, real=False)
    score = fid.compute()
    return score

def inception_score(model, n_samples, batch_size, device, disable_tqdm=True):
    is_metric = InceptionScore(normalize=True).to(device)
    n_batches, remainder = divmod(n_samples, batch_size)
    checked = False
    for _ in tqdm(range(n_batches), desc="Generating samples for inception score"):
        samples = model.sample(batch_size, disable_tqdm=disable_tqdm)
        if not checked:
            _validate_sample_assumptions(samples, device, "generated samples")
            checked = True
        is_metric.update(samples)
    if remainder:
        samples = model.sample(remainder)
        if not checked:
            _validate_sample_assumptions(samples, device, "generated samples")
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
        "--data-dir",
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, train_index = load_model(args.checkpoint, device)

    if args.metric == "is":
        score_mean, score_std = inception_score(model, args.n_samples, args.batch_size, device)
        print(f"{score_mean} +- {score_std}")
    else: # FID
        dataset_name, _ = get_dataset_and_model_from_path(args.checkpoint)
        dataset = load_dataset(
            dataset_name,
            data_dir=args.data_dir,
            transform=None,
            random_horizontal_flip=False,
        )
        non_train_index = torch.tensor(sorted(set(range(len(dataset))) - set(train_index.tolist())), dtype=torch.long)
        n_samples = min(args.n_samples, non_train_index.shape[0])
        samples = non_train_index[:n_samples]
        dataset = Subset(dataset, samples)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        score = fid_score(model, data_loader, device)
        print(f"FID: {score}")
