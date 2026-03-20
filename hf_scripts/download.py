from pathlib import Path
import argparse

from huggingface_hub import get_token, snapshot_download

try:
    from hf_utils import authenticate_hf
except ModuleNotFoundError:
    from hf_scripts.hf_utils import authenticate_hf

HF_TOKEN = get_token()
REPO_ID = "Malassila/eb-mia"
REPO_TYPE = "model"


def download_directory(
    local_dir,
    remote_dir,
    file_pattern=None,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
):
    '''
    Download a directory or matching files from a Hugging Face repo.
    Args:
        local_dir (str | Path): Local directory to download into.
        remote_dir (str): Directory in the remote repo.
        file_pattern (str | None): Optional file pattern relative to `remote_dir`.
        repo_id (str): Hugging Face repository id.
        repo_type (str): Hugging Face repository type.
    Returns:
        str: Local snapshot path returned by Hugging Face Hub.
    '''
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = str(remote_dir).strip("/")
    allow_patterns = f"{remote_dir}/**" if file_pattern is None else f"{remote_dir}/{file_pattern.lstrip('/')}"
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )


def parse_args():
    '''
    Parse CLI arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Download a repo directory from Hugging Face.")
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory to download into.",
    )
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Source directory path in the Hugging Face repo.",
    )
    parser.add_argument(
        "--file-pattern",
        default=None,
        help="Optional glob pattern (relative to --remote-dir) to download matching files only, e.g. '*-epoch10.pth'.",
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
    '''
    Run the download CLI.
    Returns:
        None
    '''
    args = parse_args()
    authenticate_hf(token=HF_TOKEN)
    local_path = download_directory(
        local_dir=args.local_dir,
        remote_dir=args.remote_dir,
        file_pattern=args.file_pattern,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )
    print("Downloaded to:", local_path)


if __name__ == "__main__":
    main()
