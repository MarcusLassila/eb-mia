from pathlib import Path
import argparse

from huggingface_hub import get_token, upload_folder

from hf_utils import authenticate_hf

HF_TOKEN = get_token()
REPO_ID = "Malassila/eb-mia"
REPO_TYPE = "model"
DEFAULT_MODELS_DIR = Path("trained_models")


def upload_models(
    local_dir=DEFAULT_MODELS_DIR,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    remote_dir=None,
):
    '''Upload model checkpoints to Hugging Face. Args: local_dir (str|Path), repo_id (str), repo_type (str), remote_dir (str|None). Returns: str.'''
    local_dir = Path(local_dir)
    if remote_dir is None:
        remote_dir = local_dir.as_posix()
    if not local_dir.exists():
        raise FileNotFoundError(f"models directory not found: {local_dir}")
    return upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=remote_dir,
    )


def parse_args():
    '''Parse CLI arguments. Args: None. Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Upload model checkpoints to Hugging Face.")
    parser.add_argument(
        "--models-dir",
        nargs="?",
        default=str(DEFAULT_MODELS_DIR),
        help="Folder with model checkpoints to upload (default: trained_models).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    authenticate_hf(token=HF_TOKEN)
    commit_url = upload_models(local_dir=args.models_dir, remote_dir=Path(args.models_dir).as_posix())
    print("Uploaded to:", commit_url)
