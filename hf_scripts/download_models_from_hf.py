from pathlib import Path
import argparse

from huggingface_hub import get_token, snapshot_download

from hf_utils import authenticate_hf

HF_TOKEN = get_token()
REPO_ID = "Malassila/eb-mia"
REPO_TYPE = "model"
DEFAULT_MODELS_DIR = Path("trained_models")


def download_models(
    models_dir=DEFAULT_MODELS_DIR,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
):
    '''Download model checkpoints from Hugging Face. Args: models_dir (str|Path), repo_id (str), repo_type (str). Returns: str.'''
    models_dir = Path(models_dir)
    remote_dir = models_dir.as_posix()
    local_root = Path(".")
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_root),
        allow_patterns=f"{remote_dir}/**",
    )


def parse_args():
    '''Parse CLI arguments. Args: None. Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Download model checkpoints from Hugging Face.")
    parser.add_argument(
        "models_dir",
        nargs="?",
        default=str(DEFAULT_MODELS_DIR),
        help="Folder with model checkpoints to download (default: trained_models).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    authenticate_hf(token=HF_TOKEN)
    local_dir = download_models(models_dir=args.models_dir)
    print("Downloaded to:", local_dir)
