from pathlib import Path
import argparse

from huggingface_hub import get_token, upload_folder

try:
    from hf_utils import authenticate_hf
except ModuleNotFoundError:
    from hf_scripts.hf_utils import authenticate_hf

HF_TOKEN = get_token()
REPO_ID = "Malassila/eb-mia"
REPO_TYPE = "model"


def upload_directory(
    local_dir,
    remote_dir,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
):
    '''Upload a local directory to a directory in a Hugging Face repo. Args: local_dir (str|Path), remote_dir (str), repo_id (str), repo_type (str). Returns: str.'''
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"local directory not found: {local_dir}")
    return upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=remote_dir,
    )


def parse_args():
    '''Parse CLI arguments. Args: None. Returns: argparse.Namespace.'''
    parser = argparse.ArgumentParser(description="Upload a local directory to Hugging Face.")
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory to upload.",
    )
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Target directory path in the Hugging Face repo.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Hugging Face repo id (default: {REPO_ID}).",
    )
    parser.add_argument(
        "--repo-type",
        default=REPO_TYPE,
        help=f"Hugging Face repo type (default: {REPO_TYPE}).",
    )
    return parser.parse_args()


def main():
    '''Run upload CLI. Args: None. Returns: None.'''
    args = parse_args()
    authenticate_hf(token=HF_TOKEN)
    commit_url = upload_directory(
        local_dir=args.local_dir,
        remote_dir=args.remote_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )
    print("Uploaded to:", commit_url)


if __name__ == "__main__":
    main()
